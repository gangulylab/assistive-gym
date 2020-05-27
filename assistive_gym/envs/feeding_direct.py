from .feeding_robots import FeedingJacoEnv
import numpy as np
import pybullet as p
from numpy.linalg import norm
from copy import deepcopy
from collections import deque

class FeedingJacoDirectEnv(FeedingJacoEnv):
    """
    List of obs processing:
        rebase spoon position against original
        normalize predictions
    """
  
    scale = .1

    def step(self,action):
        obs,r,done,info = super().step(action)
        self._update_pos()

        click = norm(obs[7:10]) < 0.025
        self.click = click        

        return obs,r,done,info

    def reset(self):
        obs = super().reset()

        self.click = False

        # self.real_step = [True]

        self._update_pos()
        
        org_spoon_pos = deepcopy(self.spoon_pos)
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.id)
        start = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
             basePosition=org_spoon_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        self.pred_visual = deque([-1]*10,10)

        return obs

    def oracle2trajectory(self,oracle_action):
        old_spoon_pos = self.spoon_pos
        real_state = p.saveState(self.id)
        super().step(oracle_action)
        self._update_pos()
        new_spoon_pos = self.spoon_pos
        p.restoreState(real_state)
        return new_spoon_pos - old_spoon_pos


    def target2obs(self, pred_target, obs):
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[250, 110, 0, 1], physicsClientId=self.id)
        new_pred_visual = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
             basePosition=pred_target, useMaximalCoordinates=False, physicsClientId=self.id)
        p.removeBody(self.pred_visual[0])
        self.pred_visual.append(new_pred_visual)

        # print(norm(self.target_pos - pred_target))

        return np.concatenate((obs[0], self.spoon_pos-pred_target, obs[1]))

    def _update_pos(self):
        self.spoon_pos = np.array(p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)[0])
