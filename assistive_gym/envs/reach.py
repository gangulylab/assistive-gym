import os
from gym import spaces
import numpy as np
import pybullet as p

from itertools import product

from .env import AssistiveEnv

class ReachEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco', reward_weights=[1,.01]):
		super(ReachEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=21)
		self.observation_space = spaces.Box(-np.inf,np.inf,(17,), dtype=np.float32)
		self.og_init_pos = np.array([-0.5, 0, 0.8])
		self.reward_weights = reward_weights
		self.num_episodes = -1

	def step(self, action):
		# reward_distance = np.linalg.norm(self.target_pos - self.tool_pos)
		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		obs = self._get_obs()

		new_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		reward_distance = -new_dist # Penalize distances away from target
		reward_action = -np.linalg.norm(action) # Penalize actions
		if new_dist < .025:
			self.task_success += 1

		reward = np.dot([1,.01],[reward_distance, reward_action]) + 100*(new_dist < .025)

		info = {
			'task_success': self.task_success,
			'distance_target': new_dist,
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

		robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, tool_pos - self.target_pos, robot_joint_positions]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.num_episodes += 1
		if self.num_episodes == 0:
			self._reset()
		
		self.task_success = 0
		self.init_pos = np.array(p.getLinkState(self.robot, 11, computeForwardKinematics=True, physicsClientId=self.id)[0])

		p.removeBody(self.target)
		# target_pos = self.target_pos = np.array([-.3,-.6,.85])
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[.9,.6,.35], rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		sphere_collision = -1
		target_pos = self.target_pos = np.array([-.3,-.6,.85])+np.array([.9,.6,.35])*self.np_random.uniform(-1,1,3)
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		return self._get_obs()


	def _reset(self):
		self.setup_timing()
		_human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender\
			 = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False, static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		target_pos = self.target_pos = np.array([0,-1,1])
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		init_pos = self.init_pos = self.init_start_pos()
		init_pos = self.init_pos = np.array([1,-1.2,1.2])
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 11, init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])

		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)

		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)

		# Enable rendering
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

	def update_targets(self):
		pass

	def init_start_pos(self):
		t = 0
		return (1-t)*(self.og_init_pos + self.np_random.uniform(-.02,.02,3)) + t*self.target_pos

	def oracle2trajectory(self,oracle_action):
		old_tool_pos = self.tool_pos
		real_state = p.saveState(self.id)
		super().step(oracle_action)
		new_tool_pos = self.tool_pos
		p.restoreState(real_state)
		return new_tool_pos - old_tool_pos

	def target2obs(self, pred_target, obs):
		return np.concatenate((obs[0], self.tool_pos-pred_target, obs[1]))

	@property
	def tool_pos(self):
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

class ReachJacoEnv(ReachEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
