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
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Infrastructure.ClusterJob import SolverJob
from PyFoam.RunDictionary.SolutionFile import SolutionFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.ParallelExecution import LAMMachine
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Error import error

import pyvista as vtki

import time
from time import gmtime, strftime
import logging
import warnings

import sys
import subprocess
from subprocess import Popen, PIPE, CalledProcessError

solver = 'simpleFoam'
solveroptions='-case'
origcase = 'baseCase'

#logger_g = logging.getLogger(__name__)
#logging.basicConfig(
#        filename='foamstatus.log',
#        format='%(asctime)s | %(message)s',
#        datefmt='%m/%d/%Y %I:%M:%S %p',
#        level=logging.INFO)

#%%
class turb_model_parameters(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 30}
    
    # Initialize the parameters for the Lorenz system and the RL
    def __init__(self,env_config):
        # remove old log files
        with open(f'{origcase}/logr.remove', 'wb') as fp:
            subprocess.run(
                f'rm {origcase}/log.* {origcase}/*.txt',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
            
        # remove old solution directories
        with open(f'{origcase}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {origcase}/0.* {origcase}/[1-9]*',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
        
        # remove old solution directories
        with open(f'{origcase}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {origcase}/VTK',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
        
        # remove old solution directories
#        subprocess.run(
#            f'rm -r postProcessing',
#            shell=True,
#            stderr=subprocess.STDOUT
#        )
        if env_config['test']:
            self.worker_index = env_config['worker_index']
        else:
            self.worker_index = env_config.worker_index
            
        self.number_steps = 0
        self.end_time = env_config['end_time']
        self.write_interval = env_config['write_interval']
        self.a1 = None
        self.vx = env_config['vx']
        self.h = env_config['h']
        
        self.res_ux_tol = env_config['res_ux_tol']
        self.res_uy_tol = env_config['res_uy_tol']
        self.res_p_tol = env_config['res_p_tol']
        self.res_k_tol = env_config['res_k_tol']
        self.res_eps_tol = env_config['res_eps_tol']
        
        self.reward_type = env_config['reward_type']
        self.states_type = env_config['states_type']
        
        data = np.loadtxt('./baseCase/cf_expt.csv',delimiter=',',skiprows=0)
        self.x_expt = data[:,0]
        self.cf_expt = data[:,1]

        self.casename = 'baseCase_'+str(self.worker_index)
        self.csvfile = 'progress_'+str(self.worker_index)+'.csv'  

        self.state = None
        
        self.num_parameters = env_config['num_parameters']    
        self.actions = np.zeros(self.num_parameters)
        self.count_parameters = 0
        
        self.a1 = env_config['a1']
        self.reward = env_config['reward']                
        self.actions_low = env_config['actions_low']
        self.actions_high = env_config['actions_high']

        # Lower bound for relaxation factors
        turb_model_low = np.array([-1.0]) #env_config['a1_low'] #np.array([a1_low])        
        
        # Upper bound for relaxation factors
        turb_model_high  = np.array([1.0]) #env_config['a1_high'] #np.array([a1_high])
        
        #Define the bounded action space
        self.action_space = spaces.Box(turb_model_low, turb_model_high, dtype=np.float64)
        
        # Define the unbounded state-space
        if self.states_type == 1:
            high1 = np.array([np.finfo(np.float64).max])
        elif self.states_type == 2:
            high1 = np.array([np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max])
    
        self.observation_space = spaces.Box(-high1, high1, dtype=np.float64)
        
        # penalty for divergence
        self.high_penalty = 10
        
        self.seed()
            
    # Seed for random number generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset the environment at the end of each episode
    def reset(self):        
        pid = self.worker_index #os.getpid()
#        logger_g.info(f'{pid}')
        
        self.casename = 'baseCase_'+str(pid)
        self.csvfile = 'progress_'+str(pid)+'.csv'  
        orig = SolutionDirectory(origcase,archive=None,paraviewLink=False)
        case = orig.cloneCase(self.casename )
        
        control_dict = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
        control_dict["endTime"] = self.end_time
        control_dict["writeInterval"] = self.write_interval
        control_dict.writeFile()
            
        # remove old log files
        with open(f'{self.casename}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm {self.casename}/log.*',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
            
        # remove old solution directories
        with open(f'{self.casename}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {self.casename}/0.* {self.casename}/[1-9]*',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
        
        # remove old solution directories
        with open(f'{self.casename}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {self.casename}/VTK',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
        
        # remove old solution directories
#        subprocess.run(
#            f'rm -r postProcessing',
#            shell=True,
#            stderr=subprocess.STDOUT
#        )
        
        self.count_parameters = 0
        if self.states_type == 1:
            #self.actions[self.count_parameters] = self.a1[0]
            self.state = np.array([self.a1[0]]) 
        elif self.states_type == 2:
            self.state = np.hstack((self.a1,self.reward))
                    
        return np.array(self.state)
    
    # Update the state of the environment
    def step(self, action):
        
        assert self.action_space.contains(action) , "%r (%s) invalid" %(action, type(action))
        
        pid = self.worker_index #os.getpid()
        logger_m = logging.getLogger(__name__)
        logging.basicConfig(filename=f'foamstatus_{pid}.log',
                            format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)
        
        done = False
        
        self.actions[self.count_parameters] = action
        reward = 0
        itercount = 0
        res_ux,res_uy,res_p,res_k,res_eps = 1.0,1.0,1.0,1.0,1.0 
        actions_scaled = np.zeros(12)
        cf_probe_expt = np.zeros(16)
        
        if self.count_parameters == self.num_parameters - 1:
            turb_model = ParsedParameterFile(path.join(self.casename,"constant", "turbulenceProperties"))
            actions_scaled = self.actions_low + (self.actions_high - self.actions_low)*(1.0+self.actions)/2.0
#            print(actions_scaled)
            print(self.actions)
            turb_model["RAS"]["kOmegaSSTCoeffs"]["alphaK1"] = actions_scaled[0]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["alphaK2"] = actions_scaled[1]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["alphaOmega1"] = actions_scaled[2]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["alphaOmega2"] = actions_scaled[3]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["beta1"] = actions_scaled[4]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["beta2"] = actions_scaled[5]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["betaStar"] = actions_scaled[6]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["gamma1"] = actions_scaled[7]
            turb_model["RAS"]["kOmegaSSTCoeffs"]["gamma2"] = actions_scaled[8]
            turb_model.writeFile()
        
    #        logger_m.info(f'{self.worker_index}')
            logger_m.info(f'{self.a1[0]} coefficient assigned for {pid}')
            
            now = strftime("%m.%d.%Y-%H.%M.%S", gmtime())
            solverLogFile= f'log.{solver}-{now}'
            
            # to run on theta       
            proc = subprocess.Popen([f'$FOAM_APPBIN/{solver} {solveroptions} {self.casename} >> {self.casename}/{solverLogFile}'],
                            shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
                    
            proc.wait()
            (stdout, stderr) = proc.communicate()
            
            if proc.returncode != 0:
                done = True
                reward = -self.high_penalty
                res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
                logger_m.info(f'{self.a1[0]} coefficient solution diverged in the first step on {pid}')
            else:
                with open(f'{self.casename}/logr.foamLog', 'a') as fp:   
                    subprocess.run(
                            f'./extract_residual.sh {self.casename}/{solverLogFile} {self.casename} {pid}',
                            shell=True,
                            stdout = fp,
                            stderr=subprocess.STDOUT
                        )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        log_history = np.genfromtxt(f'./{self.casename}/residual_{pid}.txt')
                    
                    # if log_histyory is empty then 
                    if log_history.size == 0:
                        done = True
                        reward = -self.high_penalty
                        res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
                        logger_m.info(f'{self.a1[0]} coefficient solution no residual history')
                    else:
                        log_history = np.reshape(log_history,[-1,6])
                        itercount = int(log_history[-1,0])
                        res_ux = log_history[-1,1]
                        res_uy = log_history[-1,2]
                        res_p = log_history[-1,3]
                        res_k = log_history[-1,4]
                        res_eps = log_history[-1,5]
    
            proc = subprocess.Popen([f'$FOAM_APPBIN/foamToVTK {solveroptions} {self.casename} >> {self.casename}/logr.vtkoutput'],
                                shell=True,
                                stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    
            proc.wait()
            (stdout, stderr) = proc.communicate()
            
            # solution has blown up, caught while converting the solution to VTK format
            if proc.returncode != 0:
                done = True
                reward = -self.high_penalty
                res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
                logger_m.info(f'{self.a1[0]} coefficient solution diverged with large magnitude of velocity on {pid}')
                
            if (res_ux < self.res_ux_tol and res_uy < self.res_uy_tol and res_p < self.res_p_tol and \
                res_k < self.res_k_tol and res_eps < self.res_eps_tol) or itercount == self.end_time:    
                
                logger_m.info(f'{self.a1[0]} coefficient solution converged in {itercount} iterations')
                
                done = True
                lowerWall = vtki.PolyData(f'./{self.casename}/VTK/lowerWall/lowerWall_{itercount}.vtk')
                
                points = lowerWall.points
                points = np.array(points)
                wss_points = lowerWall.point_arrays['wallShearStress']
                wss_points = np.array(wss_points)
                
                back_slice = points[:,2] == 0 
                back_points = points[back_slice]
                wss_back_points = wss_points[back_slice]
                
                lowerwall_slice = back_points[:,1] == 0 
                back_lw_points = back_points[lowerwall_slice]
                wss_back_lw_points = wss_back_points[lowerwall_slice]
                
                cf_vtk = -2.0*wss_back_lw_points[:,0]/(self.vx**2)
                x_vtk = back_lw_points[:,0]/self.h
                
                x_slicing = self.x_expt >= 1
                self.x_expt = self.x_expt[x_slicing]
                self.cf_expt = self.cf_expt[x_slicing]
                
                cf_probe_expt = np.interp(self.x_expt,x_vtk,cf_vtk)
                
                if self.reward_type == 1:
                    reward = -np.linalg.norm(self.cf_expt - cf_probe_expt)*1
                elif self.reward_type == 2:
                    ones = np.ones(self.cf_expt.shape[0])
                    relative_error_trial = self.cf_expt/cf_probe_expt
                    relative_error_trial = (relative_error_trial - ones)**2
                    relative_error_trial[relative_error_trial>1.0] = 1.0
                    reward = -np.sum(relative_error_trial)
                            
        row = np.array([self.worker_index,self.vx,itercount,
                        res_ux,res_uy,res_p,res_k,res_eps,reward])
    
        row = np.concatenate((row,self.state))
        row = np.concatenate((row,action))
        row = np.concatenate((row,actions_scaled))
        row = np.concatenate((row,cf_probe_expt))
        
        with open(self.csvfile, 'a') as csvfile:
            np.savetxt(csvfile, np.reshape(row,[1,-1]), delimiter=",")
        
        self.reward = np.array([reward])        
        if self.states_type == 1:
            self.state = np.array([self.actions[self.count_parameters]]) 
        elif self.states_type == 2:
            self.state = np.hstack((self.a1,self.reward)) 
        
        self.count_parameters = self.count_parameters + 1
                                
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
    env_config['write_interval'] = 500 # write interval for states to be computed 
    env_config['end_time'] = 2000 # maximum number of time steps
    env_config['vx'] = 44.2
    env_config['h'] = 0.0127
    env_config['test'] = True
    env_config['worker_index'] = 1 #os.getpid()
    kosst_params = np.array([0.85,1.0,0.5,0.856,0.075,0.0828,0.09,0.5555,0.44])
    kosst_params_scaled = np.zeros(12)
    reward_base = np.array([-1.4513])
    env_config['num_parameters'] = 9
    env_config['a1'] = np.array([0.31])
    env_config['reward'] = reward_base    
    env_config['actions_low'] = np.array([0.0,0.0,0.0,0.0,0.012,0.012,0.029,0.0,0.0])
    env_config['actions_high'] = np.array([1.0,1.0,1.0,1.0,0.23,0.23,0.20,1.1,1.1])
    env_config['res_ux_tol'] =  5.0e-4
    env_config['res_uy_tol'] =  5.0e-4
    env_config['res_p_tol'] =  5.0e-2
    env_config['res_k_tol'] =  1.0e-3
    env_config['res_eps_tol'] =  1.0e-3
    env_config['reward_type'] = 2 # 1: terminal, 2: at each time step
    env_config['states_type'] = 1 # 1: single state, 2: k states history
    dp = turb_model_parameters(env_config)   
    
    done = False

    actions_a1 = 0.95*kosst_params

    start = time.time()
    reward = 0
    for i in range(1):
        k = 0
        state_0 = dp.reset()
        print(k, ' ', state_0, ' ', reward, ' ', done)  
        while not done:
            k = k + 1
            action = np.array([0.0]) #np.array([kosst_params[k]])
            state, reward, done, dict_empty = dp.step(action)
            print(k, ' ', state, ' ', reward, ' ', done)
        
        print('CPU Time = ', time.time() - start)
        done = False
        
        