"""
Created on Tue Jun  2 13:36:55 2020

@author: suraj

Custom environment to update the relaxationfactor for OpenFoam simulation.
OpenFoam simulation is for backward facing step with velocity inlet as the 
boundary condition

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

from os import path
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector

from time import gmtime, strftime

solver = 'simpleFoam'
solveroptions=""

import sys
import subprocess
from subprocess import Popen, PIPE, CalledProcessError

#%%
class dynamic_parameters(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 30}
    
    # Initialize the parameters for the Lorenz system and the RL
    def __init__(self,env_config):
        # remove old log files
        subprocess.run(
            f'rm log.*',
            shell=True,
            stderr=subprocess.STDOUT
        )
        
        # remove old solution directories
        subprocess.run(
            f'rm -r 0.* [1-9]*',
            shell=True,
            stderr=subprocess.STDOUT
        )
        
        # remove old solution directories
#        subprocess.run(
#            f'rm -r postProcessing',
#            shell=True,
#            stderr=subprocess.STDOUT
#        )
        
        self.csvfile = 'progress.csv'                
        self.number_steps = 0
        self.write_frequency = env_config['write_frequency']
        self.max_steps = env_config['max_steps']
        self.state = None
        
        write_interval = ParsedParameterFile(path.join("system", "controlDict"))
        write_interval["writeInterval"] = self.write_frequency
        write_interval.writeFile()
        
        number_steps = ParsedParameterFile(path.join("system", "controlDict"))
        number_steps["endTime"] = self.max_steps
        number_steps.writeFile()
        
        relax_p_low = 0.4
        relax_u_low = 0.4
        relax_p_high = 0.7
        relax_u_high = 0.7

        # Lower bound for relaxation factors
        relaxation_low = np.array([relax_p_low,relax_u_low])        
        
        # Upper bound for relaxation factors
        relaxation_high = np.array([relax_p_high,relax_u_high])
        
        #Define the bounded action space
        self.action_space = spaces.Box(relaxation_low, relaxation_high, dtype=np.float64)
        
        # Define the unbounded state-space
        self.vx_low = 4.0
        self.vx_high = 8.0
        
        velocity_low = np.array([self.vx_low])
        velocity_high = np.array([self.vx_high])
        
        self.observation_space = spaces.Box(velocity_low, velocity_high, dtype=np.float64)

        # Define the unbounded state-space
        high1 = np.array([np.finfo(np.float32).max])
#        self.observation_space = spaces.Box(-high1, high1, dtype=np.float64)
        
        # Stepwise rewards
        self.negative_reward = -10
        self.positive_reward = 10
        
        self.seed()
            
    # Seed for random number generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset the environment at the end of each episode
    def reset(self):        
        # remove old log files
        subprocess.run(
            f'rm log.*',
            shell=True,
            stderr=subprocess.STDOUT
        )
        
        # remove old solution directories
        subprocess.run(
            f'rm -r 0.* [1-9]*',
            shell=True,
            stderr=subprocess.STDOUT
        )
        
        # remove old solution directories
#        subprocess.run(
#            f'rm -r postProcessing',
#            shell=True,
#            stderr=subprocess.STDOUT
#        )
        
        self.state = np.array([7.33]) #self.np_random.uniform(low=self.vx_low, high=self.vx_high, size=(1,))
        
        return np.array(self.state)
    
    # Update the state of the environment
    def step(self, action):
        # set up the case with random velicity condition between u = 4 to 8 m/s
        
        vx = self.state[0]
        velBC = ParsedParameterFile(path.join("0", "U"))
        velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(vx,0,0))
        velBC.writeFile()
        
        assert self.action_space.contains(action) , "%r (%s) invalid" %(action, type(action))
        
        done = False
        
        # numerical schemes relaxation parameters (action)
        relax_p, relax_u = action 
        
        relaxP = ParsedParameterFile(path.join("system", "fvSolution"))
        relaxP["relaxationFactors"]["fields"]["p"] = relax_p
        relaxP.writeFile()
        
        #relaxU = ParsedParameterFile(path.join(templateCase.name,"system", "fvSolution"))
        relaxU = ParsedParameterFile(path.join("system", "fvSolution"))
        relaxU["relaxationFactors"]["equations"]["U"] = relax_u
        relaxU.writeFile()
        
        now = strftime("%m.%d.%Y-%H.%M.%S", gmtime())
        solverLogFile= f'log.{solver}-{now}'
        
        # to run on theta
    #    subprocess.run(
    #            f'$FOAM_APPBIN/{solver} {solveroptions} >> {solverLogFile}',
    #            shell=True,
    #            check=True,
    #            stdout=fp,
    #            stderr=subprocess.STDOUT
    #        )
        
        subprocess.run(
                f'{solver} {solveroptions} >> {solverLogFile}',
                shell=True,
                check=True,
                stderr=subprocess.STDOUT
            )
        
        file = open(f'{solverLogFile}', 'r')
        lines = file.read().splitlines()
        line = lines[-9]
        columns = [col.strip() for col in line.split(' ') if col]
        check = columns[-1]
        
        if check != 'iterations': 
            line = lines[-15]
            columns = [col.strip() for col in line.split(' ') if col]
            res_ux = np.float64(columns[7][:-1])
            line = lines[-14]
            columns = [col.strip() for col in line.split(' ') if col]
            res_uy = np.float64(columns[7][:-1])
            line = lines[-13]
            columns = [col.strip() for col in line.split(' ') if col]
            res_p = np.float64(columns[7][:-1])
            line = lines[-17]
            columns = [col.strip() for col in line.split(' ') if col]
            itercount = np.int(columns[2])
            reward = -itercount
                
        else:
            # solution converged
            done = True
            line = lines[-18]
            columns = [col.strip() for col in line.split(' ') if col]
            res_ux = np.float64(columns[7][:-1])
            line = lines[-17]
            columns = [col.strip() for col in line.split(' ') if col]
            res_uy = np.float64(columns[7][:-1])
            line = lines[-16]
            columns = [col.strip() for col in line.split(' ') if col]
            res_p = np.float64(columns[7][:-1])
            line = lines[-20]
            columns = [col.strip() for col in line.split(' ') if col]
            itercount = np.int(columns[2])
            reward = -itercount
                
        row = np.array([vx,relax_u,relax_p,itercount,res_ux,res_uy,res_p,reward])
        with open(self.csvfile, 'a') as csvfile:
            np.savetxt(csvfile, np.reshape(row,[1,-1]), delimiter=",")
                                
        return np.array(self.state), reward, done, {}
    
    #function for rendering instantaneous figures, animations, etc. (retained to keep the code OpenAI Gym optimal)
    def render(self, mode='human'):
        self.n +=0

    # close the training session
    def close(self):
        return 0

#%% 
#-----------------------------------------------------------------------------#
# test the class step function
#-----------------------------------------------------------------------------#

if __name__ == '__main__':
        
    # create the instance of a class    
    env_config = {}
    env_config['write_frequency'] = 1000 # update frequency for relaxation factors
    env_config['max_steps'] = 5000 # maximum number of time steps
    
    dp = dynamic_parameters(env_config)   
    
    done = False
    
    # action parameters for every 200 time steps
    actions_p = [0.2,0.4,0.4,0.4,0.4,0.4]
    actions_u = [0.2,0.8,0.8,0.8,0.8,0.8]
    k = 0    
    while k < 1:
        dp.reset()
        action = [actions_p[k],actions_u[k]]
        state, reward, done, dict_empty = dp.step(action)
        if k < 5:
            k = k + 1
        print(k, ' ', state, ' ', reward)