from .laptop import LaptopJacoEnv
import numpy as np
import pybullet as p
import gym
import numpy.random as random
from numpy.linalg import norm
from copy import deepcopy

import os

LEFT,RIGHT,FORWARD,BACKWARD,UP,DOWN,NULL = 'left','right','forward','backward','up','down',None

class TestEnv:
	def __init__(self,env):
		if isinstance(env,str):
			self.env = {
				"LaptopJaco-v0": LaptopJacoEnv,
			}[env]()
		else:
			self.env = env
		self.new_pos = -1

	def step(self,action):
		print(action)
		if action not in [LEFT,RIGHT,FORWARD,BACKWARD,UP,DOWN]:
			return None,0,False,{}

		curr_pos = np.array(p.getLinkState(self.env.robot, 7, computeForwardKinematics=True, physicsClientId=self.env.id)[0])
		joint_states = p.getJointStates(self.env.robot, jointIndices=self.env.robot_left_arm_joint_indices, physicsClientId=self.env.id)
		joint_positions = np.array([x[0] for x in joint_states])
		
		new_pos = curr_pos + {
			LEFT: np.array([1,0,0]),
			RIGHT: np.array([-1,0,0]),
			BACKWARD: np.array([0,1,0]),
			FORWARD: np.array([0,-1,0]),
			DOWN: np.array([0,0,-1]),
			UP: np.array([0,0,1]),
		}[action]*.5
		new_pos = self.env.target_pos

		new_joint_positions = np.array(p.calculateInverseKinematics(self.env.robot, 11, new_pos, physicsClientId=self.env.id))
		new_joint_positions = new_joint_positions[:7]
		action = new_joint_positions - joint_positions

		p.removeBody(self.new_pos)
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.env.id)
		self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
				basePosition=new_pos, useMaximalCoordinates=False, physicsClientId=self.env.id)

		return self.env.step(action)

	def reset(self):
		obs = self.env.reset()

		# switch_pos = [0,-.5,1]
		# switch_scale = .0005
		# on_off = True
		# switch_file = 'on_switch.urdf' if on_off else 'off_switch.urdf'
		# self.light_switch = p.loadURDF(os.path.join(self.env.world_creation.directory, 'light_switch', switch_file), basePosition=switch_pos, baseOrientation=p.getQuaternionFromEuler([-np.pi/2, 0, 0], physicsClientId=self.env.id),\
		# 	 physicsClientId=self.env.id,globalScaling=switch_scale)

		return obs

	def seed(self,value):
		self.env.seed(value)

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

