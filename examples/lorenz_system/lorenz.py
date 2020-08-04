"""
Created on Tue Jun  2 13:36:55 2020

@author: suraj

Taken from supplementary material of 
"Restoring chaos using deep reinforcement learning" Chaos 30, 031102 (2020)

"""
import gym
import os
import csv
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

class lorenzEnv_transient(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 30}
    
    # Initialize the parameters for the Lorenz system and the RL
    def __init__(self, env_config):
        self.n = env_config['n']
        self.path = os.getcwd()+'/lorenz_transient_terminal.csv'
        self.t = 0
        self.max_episode_steps = 4000 # env_config['episode'] # Number of max steps in an episode
        self.explore_factor = 2000
        self.rho   =  20.0
        self.sigma = 10.0
        self.beta  = 8.0/3.0
        self.tau   = 0.02
                
        # Location of the two fix-points of the Lorenz system
        self.fp1 = np.array([-np.sqrt(self.beta*(self.rho - 1.)), -np.sqrt(self.beta*(self.rho - 1.)) , (self.rho - 1.)])
        self.fp2 = np.array([np.sqrt(self.beta*(self.rho - 1.)), np.sqrt(self.beta*(self.rho - 1.)) , (self.rho - 1.)])
        
        # Upper bound for control perturbation values
        high = np.array([self.rho/10, self.sigma/10, self.beta/10])

        # Define the unbounded state-space
        high1 = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, 
                          np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max  ])
        self.observation_space = spaces.Box(-high1, high1, dtype=np.float32)
        
        #Define the bounded action space
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state  = None
        self.reward = None
        
        # Stop episode after 4000 steps
        self.n_terminal = 4000
        
        # Stepwise rewards
        self.negative_reward = -10.0
        self.positive_reward = 10.0
        
        #Terminal reward
        self.term_punish =  self.negative_reward*10.
        self.term_mean = self.negative_reward/5.    # -2.0
        
        # To compute mean reward over the final 2000 steps
        self.n_max = self.n_terminal - self.explore_factor
        self.rewards = []
    
    # Seed for random number generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # assign random state to the environment at the begining of each episode
    def reset(self):
        def f(state, t):
            x, y, z = state
            return (self.sigma) * (y - x), x * ((self.rho) - z) - y, x * y - (self.beta) * z

        self.state = self.np_random.uniform(low=-40, high=40, size=(6,))#assign position coordinates
        self.state[3:6] = f(self.state[0:3], 0)# assign velocity following the Lorenz system

        return np.array(self.state)
    
    # Update the state of the environment
    def step(self, action):
        assert self.action_space.contains(action) , "%r (%s) invalid"%(action, type(action))
        done = False

        rhop, sigmap, betap = action #perturbation parameters (action)

        state=self.state[0:3] # position coordinates

        def f(state, t):
            x, y, z = state	# unpack the position coordinates


            return (self.sigma + sigmap) * (y - x), x * ((self.rho + rhop) - z) - y, x * y - (self.beta + betap) * z  # derivatives when continous action space


        t=np.linspace(self.t,  self.t + (self.n)*(self.tau), (self.n + 1)) # time array for single step
        state2 = odeint(f, state, t) # updated state of the Lorenz system with perturbed parameters after time marching
        xdot, ydot, zdot = f(state2[self.n,:], t) #velocity vector for the new state

        velo_mag = (np.sqrt(xdot**2 + ydot**2 + zdot**2)) # magnitude of velocity in present state
        trofal = (velo_mag<40) # check if magnitude of velocity is less than the threshold (40)
   
        #assignment of step reward to the RL agent
        if trofal:
            reward = self.negative_reward
        else:
            reward = self.positive_reward

        self.rewards.append(reward)

        #check if end of episode reached; if yes assign terminal reward, and enter new episode.
        if len(self.rewards)>=self.n_terminal:
            if np.mean(self.rewards[self.n_max:])<= self.term_mean:
                reward = self.term_punish
                self.rewards = []
                done = True


        self.state = state2[self.n,0], state2[self.n,1], state2[self.n,2], xdot, ydot, zdot # update the state vector

        row = [self.t, self.state[0], self.state[1], self.state[2], rhop, sigmap, betap, xdot, ydot, zdot, velo_mag, reward] # store the values of relevant quantities for writing them in a csv file
        

        self.t = t[-1] # update the instantaneous time

        # Check if file-name already exists; if yes, delete its contents.
#        if (self.t <= 5.*(self.tau) and  os.path.isfile(self.path)) :
        if (self.t <= 1.0*(self.tau)  and os.path.isfile(self.path)) :
            f = open(self.path, 'r+')
            f.truncate()
        #	write to file
        with open(self.path, 'a') as output:
            writer = csv.writer(output)
            writer.writerow(row)

        output.close()

        return np.array(self.state), reward, done, {}
    
    #function for rendering instantaneous figures, animations, etc. (retained to keep the code OpenAI Gym optimal)
    def render(self, mode='human'):
        self.n +=0

    # close the training session
    def close(self):
        return 0