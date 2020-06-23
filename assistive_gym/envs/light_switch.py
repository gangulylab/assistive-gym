import os
from gym import spaces
import numpy as np
import pybullet as p

from itertools import product

from .env import AssistiveEnv

class LightSwitchEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco', reward_weights=[1,.01,1,1]):
		super(LightSwitchEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=21)
		self.observation_space = spaces.Box(-np.inf,np.inf,(21,), dtype=np.float32)
		self.og_init_pos = np.array([-0.5, 0, 0.8])
		self.reward_weights = reward_weights

	def step(self, action):
		reward_distance = np.linalg.norm(self.target_pos - self.tool_pos)
		switch_pos = np.array(p.getBasePositionAndOrientation(self.switch, physicsClientId=self.id)[0])
		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		tool_force, tool_force_at_target, target_contact_pos, contact_switch_count = self.get_total_force()
		end_effector_velocity = np.linalg.norm(p.getLinkState(self.tool, 1, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
		if target_contact_pos is not None:
			target_contact_pos = np.array(target_contact_pos)
		obs = self._get_obs([tool_force])

		new_switch_pos = np.array(p.getBasePositionAndOrientation(self.switch, physicsClientId=self.id)[0])
		reward_switch_distance = -np.linalg.norm(switch_pos-new_switch_pos)
		reward_distance -= np.linalg.norm(self.target_pos - self.tool_pos) # Penalize distances away from target
		reward_action = -np.linalg.norm(action) # Penalize actions
		if target_contact_pos is not None\
			and tool_force_at_target < 10:
			self.task_success += 1

		reward = np.dot(list(self.reward_weights.values()),
			[reward_distance, reward_action,(tool_force_at_target>.5),-contact_switch_count])\
			 + 10*reward_switch_distance

		if self.gui and tool_force_at_target > 0:
			print('Task success:', self.task_success, 'tool force at target:', tool_force_at_target)

		self.contact_switch_count += contact_switch_count
		self.switch_move = np.linalg.norm(self.switch_pos - new_switch_pos)
		info = {'task_success': self.task_success,
		'distance_target': np.linalg.norm(self.target_pos - self.tool_pos),
		'switch_count': contact_switch_count,
		'switch_move': self.switch_move}
		done = False

		return obs, reward, done, info

	def get_total_force(self):
		tool_force = 0
		tool_force_at_target = 0
		target_contact_pos = None
		contact_switch_count = 0
		screen_contact_pos = None
		for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
			tool_force += c[9]
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switch, linkIndexB=0, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			# Enforce that contact is close to the target location
			if linkA in [0,1] and np.linalg.norm(contact_position - self.target_pos) < 0.025:
				tool_force_at_target += c[9]
				target_contact_pos = contact_position
			else:
				contact_switch_count += 1
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switch, linkIndexB=1, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			contact_switch_count += 1
		for c in p.getContactPoints(bodyA=self.robot, bodyB=self.switch, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			contact_switch_count += 1
		return tool_force, tool_force_at_target, target_contact_pos, contact_switch_count

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		screen_pos = np.array(p.getLinkState(self.switch, 0, computeForwardKinematics=True,physicsClientId=self.id)[0])

		robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, tool_pos - self.target_pos, robot_joint_positions, screen_pos-torso_pos, forces]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.setup_timing()
		self.task_success = 0
		self.contact_switch_count = 0
		self.switch_move = 0
		_human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender\
			 = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False, static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		# Place a switch on a wall
		wall_pos,wall_orient = [
			(np.array([0,-1.1,1]),[0,0,0,1]),
			(np.array([.6,-.2,1]),p.getQuaternionFromEuler([0, 0, np.pi/2])),
			(np.array([-1.1,-.2,1]),p.getQuaternionFromEuler([0, 0, -np.pi/2])),
			][2]
		wall_collision = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1,.05,1])
		wall_visual = p.createVisualShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		self.wall = p.createMultiBody(basePosition=wall_pos,baseOrientation=wall_orient,baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual,physicsClientId=self.id)
		
		switch_pos = np.array([0,.1,0])+np.array([.2,0,.1])*self.np_random.uniform(-1,1,3)
		switch_pos,switch_orient = p.multiplyTransforms(wall_pos, wall_orient, switch_pos, p.getQuaternionFromEuler([-np.pi/2,0,0]), physicsClientId=self.id)
		switch_scale = .0006
		on_off = True
		switch_file = 'on_switch.urdf' if on_off else 'off_switch.urdf'
		self.switch = p.loadURDF(os.path.join(self.world_creation.directory, 'light_switch', switch_file), basePosition=switch_pos, baseOrientation=switch_orient,\
			 physicsClientId=self.id,globalScaling=switch_scale)

		target_pos = np.array([0,.032,.017]) if on_off else np.array([0,-.032,.017])
		target_pos, target_orient = p.multiplyTransforms(switch_pos, switch_orient, target_pos, [0, 0, 0, 1], physicsClientId=self.id)
		self.target_pos = np.array(target_pos)
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		# self.update_targets()

		init_pos = self.init_pos = self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 8, init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
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

		return self._get_obs([0])
	
	def init_start_pos(self):
		t = 0
		return (1-t)*(self.og_init_pos + self.np_random.uniform(-.02,.02,3)) + t*self.target_pos

	def update_targets(self):
		pass

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

class LightSwitchJacoEnv(LightSwitchEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
