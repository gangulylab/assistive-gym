import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

class LightSwitchEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco'):
		super(LightSwitchEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=18)
		self.observation_space = spaces.Box(-np.inf,np.inf,(18,), dtype=np.float32)
		self.num_targets = 6

	def step(self, action):
		old_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		old_tool_pos = self.tool_pos
		old_traj = self.target_pos - old_tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		new_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		new_traj = self.tool_pos - old_tool_pos
		cos_off_course = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		
		tool_force, tool_force_at_target, target_contact_pos, bad_contact_count = self.get_total_force()
		self.bad_contact_count += bad_contact_count
		if target_contact_pos is not None:
			target_contact_pos = np.array(target_contact_pos)
		task_success = target_contact_pos is not None\
			and tool_force_at_target < 10\
			and self.bad_contact_count < 5
		self.task_success += task_success
		obs = self._get_obs([tool_force])

		reward_distance = old_dist - new_dist
		reward_action = -np.linalg.norm(action) # Penalize actions
		reward = np.dot([1,.01,1],
			[reward_distance, reward_action, task_success])

		info = {
			'task_success': self.task_success,
			'distance_target': new_dist,
			'diff_distance': reward_distance,
			'action_size': -reward_action,
			'cos_off_course': cos_off_course,
			'trajectory': new_traj,
			'old_tool_pos': old_tool_pos,
		}
		done = False

		return obs, reward, done, info

	def get_total_force(self):
		tool_force = 0
		tool_force_at_target = 0
		target_contact_pos = None
		bad_contact_count = 0
		for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
			tool_force += c[9]
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switch, linkIndexB=0, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			# Enforce that contact is close to the target location
			if linkA in [0,1] and np.linalg.norm(contact_position - self.target_pos) < 0.01:
				tool_force_at_target += c[9]
				target_contact_pos = contact_position
			else:
				bad_contact_count += 1
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.wall, linkIndexB=1, physicsClientId=self.id):
			bad_contact_count += 1
		for c in p.getContactPoints(bodyA=self.robot, bodyB=self.wall, physicsClientId=self.id):
			bad_contact_count += 1
		return tool_force, tool_force_at_target, target_contact_pos, bad_contact_count

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		switch_pos = np.array(p.getBasePositionAndOrientation(self.switch, physicsClientId=self.id)[0])

		# robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, forces]).ravel()
		robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, switch_pos-torso_pos, forces]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.task_success = 0
		self.bad_contact_count = 0

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

		"""set up target and initial robot position (objects set up with target)"""
		self.generate_target(self.np_random.choice(self.num_targets))
		# self.init_robot_arm(np.array([-.25,-.5,1])+np.array([.75,.5,.1])*self.np_random.uniform(-1,1,3))
		self.init_robot_arm(np.array([-0.2, -.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3))

		"""configure pybullet"""
		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.switch, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .6, cameraYaw=180, cameraPitch=-45, cameraTargetPosition=[0, .1, 1], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs([0])
	
	def init_start_pos(self,og_init_pos):
		"""exchange this function for curriculum"""
		return og_init_pos

	def init_robot_arm(self,og_init_pos):
		init_pos = self.init_pos = self.init_start_pos(og_init_pos)
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 11, init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def generate_target(self,index): 
		self.target_index = index
		# Place a switch on a wall
		wall_index = index % 3
		on_off = index // 3

		walls = [
			(np.array([0,-1.1,1]),[0,0,0,1]),
			(np.array([.6,-.2,1]),p.getQuaternionFromEuler([0, 0, np.pi/2])),
			(np.array([-1.1,-.2,1]),p.getQuaternionFromEuler([0, 0, -np.pi/2])),
			]
		wall_pos,wall_orient = walls[wall_index]
		wall_collision = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		wall_visual = p.createVisualShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		self.wall = p.createMultiBody(basePosition=wall_pos,baseOrientation=wall_orient,baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual,physicsClientId=self.id)

		wall_pos, wall_orient = p.getBasePositionAndOrientation(self.wall)
		switch = np.array([0,.1,0])+np.array([.05,0,.05])*self.np_random.uniform(-1,1,3)
		switch_pos,switch_orient = p.multiplyTransforms(wall_pos, wall_orient, switch, p.getQuaternionFromEuler([-np.pi/2,0,0]), physicsClientId=self.id)
		switch_scale = .0006
		switch_file = 'on_switch.urdf' if on_off else 'off_switch.urdf'
		self.switch = p.loadURDF(os.path.join(self.world_creation.directory, 'light_switch', switch_file), basePosition=switch_pos, useFixedBase=True, baseOrientation=switch_orient,\
			 physicsClientId=self.id,globalScaling=switch_scale)

		self.targets = [p.multiplyTransforms(switch_pos, switch_orient, target_pos, [0, 0, 0, 1])[0] for target_pos in [[0,-.032,.017],[0,.032,.017]]\
						for switch_pos,switch_orient in [p.multiplyTransforms(wall_pos, wall_orient, switch, p.getQuaternionFromEuler([-np.pi/2,0,0]))\
						for wall_pos,wall_orient in walls]]
		target_pos = self.target_pos = np.array(self.targets[self.target_index])
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		pass

	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class LightSwitchJacoEnv(LightSwitchEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
