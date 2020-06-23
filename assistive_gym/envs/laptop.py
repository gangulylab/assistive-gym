import os
from gym import spaces
import numpy as np
import pybullet as p

from itertools import product

from .env import AssistiveEnv

class LaptopEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco'):
		super(LaptopEnv, self).__init__(robot_type=robot_type, task='laptop', frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=21)
		self.observation_space = spaces.Box(-np.inf,np.inf,(21,), dtype=np.float32)
		self.og_init_pos = np.array([-0.5, 0, 0.8])

	def step(self, action):
		reward_distance = np.linalg.norm(self.target_pos - self.tool_pos)
		laptop_pos = np.array(p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)[0])
		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		tool_force, tool_force_at_target, target_contact_pos, contact_laptop_count = self.get_total_force()
		end_effector_velocity = np.linalg.norm(p.getLinkState(self.tool, 1, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
		if target_contact_pos is not None:
			target_contact_pos = np.array(target_contact_pos)
		obs = self._get_obs([tool_force])

		new_laptop_pos = np.array(p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)[0])
		reward_laptop_distance = -np.linalg.norm(laptop_pos-new_laptop_pos)
		reward_distance -= np.linalg.norm(self.target_pos - self.tool_pos) # Penalize distances away from target
		reward_action = -np.linalg.norm(action) # Penalize actions
		if target_contact_pos is not None\
			and tool_force_at_target < 10:
			self.task_success += 1

		reward = np.dot([1,.01,1,1],
			[reward_distance, reward_action,(tool_force_at_target>.5),-contact_laptop_count])\
			 + 10*reward_laptop_distance

		if self.gui and tool_force_at_target > 0:
			print('Task success:', self.task_success, 'tool force at target:', tool_force_at_target)

		self.contact_laptop_count += contact_laptop_count
		self.laptop_move = np.linalg.norm(self.laptop_pos - new_laptop_pos)
		info = {'task_success': self.task_success,
		'distance_target': np.linalg.norm(self.target_pos - self.tool_pos),
		'laptop_count': contact_laptop_count,
		'laptop_move': self.laptop_move,
		'action_size': -reward_action,
		}
		done = False

		return obs, reward, done, info

	def get_total_force(self):
		tool_force = 0
		tool_force_at_target = 0
		target_contact_pos = None
		contact_laptop_count = 0
		screen_contact_pos = None
		for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
			tool_force += c[9]
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.laptop, linkIndexB=0, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			# Enforce that contact is close to the target location
			if linkA in [0,1] and np.linalg.norm(contact_position - self.target_pos) < 0.025:
				tool_force_at_target += c[9]
				target_contact_pos = contact_position
			else:
				contact_laptop_count += 1
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.laptop, linkIndexB=1, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			contact_laptop_count += 1
		for c in p.getContactPoints(bodyA=self.robot, bodyB=self.laptop, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			contact_laptop_count += 1
		return tool_force, tool_force_at_target, target_contact_pos, contact_laptop_count

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		screen_pos = np.array(p.getLinkState(self.laptop, 0, computeForwardKinematics=True,physicsClientId=self.id)[0])

		robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, tool_pos - self.target_pos, robot_joint_positions, screen_pos-torso_pos, forces]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.setup_timing()
		self.task_success = 0
		self.contact_laptop_count = 0
		self.laptop_move = 0
		_human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender\
			 = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False, static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		# Place a laptop on a table
		table_pos = np.array([.0,-1.1,0])
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=table_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
		laptop_scale = 0.12
		laptop_pos = self.laptop_pos = table_pos+np.array([0,.2,.7])+np.array([.3,.1,0])*self.np_random.uniform(-1,1,3)
		self.laptop = p.loadURDF(os.path.join(self.world_creation.directory, 'laptop', 'laptop.urdf'), basePosition=laptop_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=laptop_scale)

		self.generate_target()

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

	def generate_target(self): 
		lbody_pos, lbody_orient = p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)
		self.target_button = np.array([0, 0, .05]) + np.array([.1, .15, 0])*self.np_random.uniform(-1,1,3)
		target_pos, target_orient = p.multiplyTransforms(lbody_pos, lbody_orient, self.target_button, [0, 0, 0, 1], physicsClientId=self.id)

		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.025, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		lbody_pos, lbody_orient = p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)
		target_pos, target_orient = p.multiplyTransforms(lbody_pos, lbody_orient, self.target_button, [0, 0, 0, 1], physicsClientId=self.id)
		self.target_pos = np.array(target_pos)
		p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

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

class LaptopJacoEnv(LaptopEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
