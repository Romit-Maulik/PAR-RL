import gym
import numpy as np
from gym import spaces

"""
State:
The current vector of chosen values

Action:
choose the next value
"""

class my_environment(gym.Env):

    def __init__(self, config):
    
        self.Scalar = config['Scalar']
        print('Scalar value : ', self.Scalar)
        
        self.observation_space = spaces.MultiDiscrete([ 4, 49, 49, 49, 49 ])
        self.action_space = spaces.Discrete(49)
        self.current_step = 0
        self.intvector = np.asarray([0,0,0,0,0], dtype=np.int64)
        
    def reset(self):
        self.current_step = 0
        self.intvector = np.asarray([0,0,0,0,0], dtype=np.int64)
        
        return self.intvector
        
    def _take_action(self, action):
        self.intvector[self.current_step +1] = action
        self.intvector[0] += 1
        
    def step(self, action):
    
        self._take_action(action)
        
        self.current_step += 1
        
        obs = self.intvector
        if self.current_step < 4:
            reward = 0
            done = False
        else:
            self.intvector[1:] += 12
            reward = -(self.Scalar - (self.intvector[3]*self.intvector[2]) / (self.intvector[1]*self.intvector[4]))**2
            self.intvector[1:] -= 12
            done = True
        
        return obs, reward, done , {}
        
    def render(self, mode="human", close=False):
        pass

if __name__ == '__main__':
    pass





