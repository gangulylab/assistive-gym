import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

reach_arena = (np.array([-.25,-.5,1]),np.array([.6,.4,.2]))
class ReachEnv(AssistiveEnv):
	def __init__(self, num_targets = 2, robot_type='jaco'):
		super(ReachEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(14,), dtype=np.float32)
		self.num_targets = num_targets

	def step(self, action):
		old_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		old_tool_pos = self.tool_pos
		old_traj = self.target_pos - old_tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		new_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		new_traj = self.tool_pos - old_tool_pos
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))

		task_success = new_dist < .01
		self.task_success += task_success
		obs = self._get_obs()

		reward_distance = old_dist - new_dist
		reward_action = -np.linalg.norm(action) # Penalize actions
		reward = np.dot([1,.01,1],
			[reward_distance, reward_action, task_success])

		info = {
			'task_success': self.task_success,
			'distance_to_target': new_dist,
			'diff_distance': reward_distance,
			# 'action_size': -reward_action,
			'cos_error': cos_error,
			# 'trajectory': new_traj,
			# 'old_tool_pos': old_tool_pos,
		}
		done = False

		return obs, reward, done, info

	def _get_obs(self):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		robot_obs = np.concatenate([tool_pos, tool_orient, robot_joint_positions]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.task_success = 0

		"""set up standard environment"""
		self.setup_timing()
		_human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender\
			 = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False, static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])

		"""set up target and initial robot position"""
		self.generate_target(self.np_random.choice(self.num_targets))
		self.init_robot_arm(reach_arena[0]+reach_arena[1]*self.np_random.uniform(-1,1,3))

		"""configure pybullet"""
		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .6, cameraYaw=180, cameraPitch=-45, cameraTargetPosition=[0, .1, 1], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs()
	
	def init_start_pos(self,og_init_pos):
		"""exchange this function for curriculum"""
		return og_init_pos

	def init_robot_arm(self,og_init_pos):
		init_pos = self.init_pos = self.init_start_pos(og_init_pos)
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 11, init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def set_target(self):
		self.target_pos = np.array(self.targets[self.target_index])
		return self.target_pos

	def generate_target(self,index): 
		self.target_index = index
		self.targets = reach_arena[0] + .1*self.np_random.uniform(-1,1,3) + list(product(*list(map(lambda extent:np.linspace(-extent,extent,3), reach_arena[1]))))
		self.targets = self.targets[[21,0,1,24,23,3,10,13,7,19,2,11,17,20,16,6,18,22,26,9,4,25,8,15,5,12,14]][:self.num_targets]
		target_pos = self.set_target()

		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		pass

	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class ReachJacoEnv(ReachEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
