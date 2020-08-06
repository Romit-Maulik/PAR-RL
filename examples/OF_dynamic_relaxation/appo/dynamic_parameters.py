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
class dynamic_parameters(gym.Env):
    
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
        self.update_frequency = env_config['update_frequency']
        self.write_interval = env_config['write_interval']
        self.single_velocity = env_config['single_velocity']
        self.res_ux_tol = env_config['res_ux_tol']
        self.res_uy_tol = env_config['res_uy_tol']
        self.res_p_tol = env_config['res_p_tol']
        self.res_k_tol = env_config['res_k_tol']
        self.res_eps_tol = env_config['res_eps_tol']
        self.reward_type = env_config['reward_type']
        self.states_type = env_config['states_type']
#        self.pido = env_config['pid']
        self.casename = 'baseCase_'+str(self.worker_index)
        self.csvfile = 'progress_'+str(self.worker_index)+'.csv'  
        self.max_steps = env_config['max_steps']
        self.state = None
        self.stateFrequency = int(self.update_frequency/self.write_interval)
        self.stateFolders = [(i+1)*self.write_interval-1 for i in range(self.stateFrequency)]
        self.vel_square = []
        self.itercount_prev = 0
        
#        orig = SolutionDirectory(origcase,archive=None,paraviewLink=False)
#        case=orig.cloneCase(self.casename )
#        
#        write_interval = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
#        write_interval["writeInterval"] = self.update_frequency
#        write_interval.writeFile()
        
        relax_p_low = 0.2
        relax_u_low = 0.2
        relax_p_high = 0.95
        relax_u_high = 0.95

        # Lower bound for relaxation factors
        relaxation_low = np.array([relax_p_low,relax_u_low])        
        
        # Upper bound for relaxation factors
        relaxation_high = np.array([relax_p_high,relax_u_high])
        
        #Define the bounded action space
        self.action_space = spaces.Box(relaxation_low, relaxation_high, dtype=np.float64)
        
        # Define the unbounded state-space
        self.vx = 0.0
        self.vx_low = env_config['vx_low']
        self.vx_high = env_config['vx_high']
        self.vx_all = np.array([39.83,8.0,4.0])
        
        velocity_low = np.array([self.vx_low])
        velocity_high = np.array([self.vx_high])

        # Define the unbounded state-space
        if self.states_type == 1:
            high1 = np.array([np.finfo(np.float64).max])
        elif self.states_type == 2:
            high1 = np.array([np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max])
    
        self.observation_space = spaces.Box(-high1, high1, dtype=np.float64)
        
        # penalty for divergence
        self.high_penalty = 2000
        
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
        case=orig.cloneCase(self.casename )
        
        if self.states_type == 1:
            write_interval = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
            write_interval["writeInterval"] = self.update_frequency
            write_interval.writeFile()
        elif self.states_type == 2:
            write_interval = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
            write_interval["writeInterval"] = self.update_frequency
            write_interval.writeFile()
            
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
        
        # set number of steps to 0 and start dynamic update
        self.number_steps = 0
        self.stateFolders = [(i+1)*self.write_interval-1 for i in range(self.stateFrequency)]
        self.vel_square = []
        self.itercount_prev = 0
                
        # set up the case with random velicity condition between u = 4 to 8 m/s
        if self.single_velocity:
            self.vx = np.array([self.vx_all[0]]) #self.np_random.uniform(low=self.vx_low, high=self.vx_high, size=(1,)) #self.vx_all[0] # additional velicities to be added
        else:
            self.vx = self.np_random.uniform(low=self.vx_low, high=self.vx_high, size=(1,)) #self.vx_all[0] # additional velicities to be added
        
        
        velBC = ParsedParameterFile(path.join(self.casename,"0", "U"))
        velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(self.vx,0,0))
        velBC.writeFile()
        
        # turbulence inlet parameters from experimental study         
        nu = 6.7e-7
        intensity = 0.00061
        eddy_visc = 0.009

        kinetic_energy = ParsedParameterFile(path.join(self.casename,"0", "k"))
        k = 1.5*(self.vx[0]*intensity)**2
        kinetic_energy["internalField"] = f"uniform {k}"
        kinetic_energy.writeFile()
        
        dissipation = ParsedParameterFile(path.join(self.casename,"0", "omega"))
        diss = int(k/(nu*eddy_visc))
        dissipation["internalField"] = f"uniform {diss}"
        dissipation.writeFile()
        
        # convert solution files to vtk format
        with open(f'{self.casename}/logr.vtkoutput', 'wb') as fp:
            
            subprocess.run(
                f'$FOAM_APPBIN/foamToVTK {solveroptions} {self.casename}',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )
        
        inlet = vtki.PolyData(f'./{self.casename}/VTK/inlet/inlet_0.vtk')
        Ub = inlet.cell_arrays['U']
        Ub = np.array(Ub)
        
        mesh = vtki.UnstructuredGrid(f'./{self.casename}/VTK/{self.casename}_0.vtk')
        Um = mesh.cell_arrays['U']
        Um = np.array(Um)
        Um = (np.sum(np.square(Um),axis=1))
        Ub = (np.sum(np.square(Ub),axis=1))
        U2 = np.average(Ub) + np.average(Um)
        
        if self.states_type == 1:
            self.state = np.array([U2]) 
        elif self.states_type == 2:
            self.state = np.array([U2 for i in range(5)])
            
        logger_m = logging.getLogger(__name__)
        logging.basicConfig(filename=f'foamstatus_{pid}.log',
                            format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)
        
#        logger_m.info(f'{self.worker_index}')
        logger_m.info(f'{self.vx[0]} velocity assigned for {pid}')
        
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
        
        # numerical schemes relaxation parameters (action)
        relax_p, relax_u = action 
        
        relaxP = ParsedParameterFile(path.join(self.casename,"system", "fvSolution"))
        relaxP["relaxationFactors"]["fields"]["p"] = relax_p
        relaxP.writeFile()
        
        #relaxU = ParsedParameterFile(path.join(templateCase.name,"system", "fvSolution"))
        relaxU = ParsedParameterFile(path.join(self.casename,"system", "fvSolution"))
        relaxU["relaxationFactors"]["equations"]["U"] = relax_u
        relaxU.writeFile()
        
        self.number_steps =  self.number_steps + self.update_frequency
        number_steps = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
        number_steps["endTime"] = self.number_steps
        number_steps.writeFile()
        
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
#            print(stderr)
            done = True
            itercount = self.high_penalty
            reward = -itercount
            res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
            logger_m.info(f'{self.vx[0]} velocity solution diverged in the first step on {pid}')
        else:
            with open(f'{self.casename}/logr.foamLog', 'a') as fp:   
#                pid = os.getpid()
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
                    itercount = 2.0*self.high_penalty
                    reward = -itercount
                    res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
                    logger_m.info(f'{self.vx[0]} velocity solution no residual history')
                else:
                    log_history = np.reshape(log_history,[-1,6])
                    itercount = int(log_history[-1,0])
                    res_ux = log_history[-1,1]
                    res_uy = log_history[-1,2]
                    res_p = log_history[-1,3]
                    res_k = log_history[-1,4]
                    res_eps = log_history[-1,5]
                    
                    if self.reward_type == 1: # assign only terminal reward
                        if res_ux < self.res_ux_tol and res_uy < self.res_uy_tol and res_p < self.res_p_tol and res_k < self.res_k_tol and res_eps < self.res_eps_tol:
                            reward = -itercount
                            done = True
                            logger_m.info(f'{self.vx[0]} velocity solution converged with {itercount} iterations on {pid}')
                        else:
                            reward = 0
                            
                    elif self.reward_type == 2: # assign reward for each time step of RL
                        if res_ux < self.res_ux_tol and res_uy < self.res_uy_tol and res_p < self.res_p_tol and res_k < self.res_k_tol and res_eps < self.res_eps_tol:
                            reward = -(itercount - self.itercount_prev)
                            self.itercount_prev = itercount
                            done = True
                            logger_m.info(f'{self.vx[0]} velocity solution converged with {itercount} iterations on {pid}')
                        else:
                            reward = -(itercount - self.itercount_prev)
                            self.itercount_prev = itercount

#        prev_folder = self.number_steps - self.update_frequency
#        if prev_folder != 0:
#            with open(f'{self.casename}/logr.remove', 'a') as fp:
#                subprocess.run(
#                    f'rm -r {self.casename}/{prev_folder}',
#                    shell=True,
#                    stdout=fp,
#                    stderr=subprocess.STDOUT
#                )
        
        proc = subprocess.Popen([f'$FOAM_APPBIN/foamToVTK {solveroptions} {self.casename} >> {self.casename}/logr.vtkoutput'],
                            shell=True,
                            stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)

        proc.wait()
        (stdout, stderr) = proc.communicate()
        
        # solution has blown up, catched while converting the solution to VTK format
        if proc.returncode != 0:
            done = True
            itercount = self.high_penalty
            reward = -itercount
            res_ux,res_uy,res_p,res_k,res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
            logger_m.info(f'{self.vx[0]} velocity solution diverged with large magnitude of velocity on {pid}')
            
        if itercount != self.high_penalty:    
            inlet = vtki.PolyData(f'./{self.casename}/VTK/inlet/inlet_{itercount}.vtk')
            Ub = inlet.cell_arrays['U']
            Ub = np.array(Ub)
            
            mesh = vtki.UnstructuredGrid(f'./{self.casename}/VTK/{self.casename}_{itercount}.vtk')
            Um = mesh.cell_arrays['U']
            Um = np.array(Um)
            Um_max = np.max(Um)
            
            if Um_max > 1.0e5:
                # velocity diverged
                done = True
                itercount = self.high_penalty
                reward = -itercount
                res_ux,res_uy,res_p, res_k, res_eps = 1.0, 1.0, 1.0, 1.0, 1.0
                if self.states_type == 1:
                    U2 = self.high_penalty 
                    self.state = np.array([U2])
                elif self.states_type == 2:
                    U2 = [self.high_penalty for i in range(self.stateFrequency)]
                    self.state = np.array(U2)
                    logger_m.info(f'{self.vx[0]} velocity solution diverged with 1e3 magnitude of velocity on {pid}')
            else:
                if self.states_type == 1:
                    Um = (np.sum(np.square(Um),axis=1))
                    Ub = (np.sum(np.square(Ub),axis=1))
                    U2 = np.average(Ub) + np.average(Um)
                    self.state = np.array([U2])
                elif self.states_type == 2:
                    if done != True:
                        folderName =  self.number_steps - self.update_frequency
                        data = np.genfromtxt(f'./{self.casename}/postProcessing/squareU/{folderName}/squareU.dat')
                        U2 = data[self.stateFolders,1] + self.vx[0]**2
                        self.state = np.array(U2) 
        else:
            if self.states_type == 1:
                U2 = self.high_penalty 
                self.state = np.array([U2])
            elif self.states_type == 2:
                U2 = [self.high_penalty for i in range(self.stateFrequency)]
                self.state = np.array(U2)
            
        
#        self.stateFolders = self.stateFolders + self.update_frequency
                
        row = np.array([self.worker_index,self.vx[0],relax_u,relax_p,itercount,
                        res_ux,res_uy,res_p,res_k,res_eps,reward])
    
        row = np.concatenate((row,self.state))
        
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
    env_config['update_frequency'] = 100 # update frequency for relaxation factors
    env_config['write_interval'] = 10 # write interval for states to be computed 
    env_config['max_steps'] = 5000 # maximum number of time steps
    env_config['single_velocity'] = True #use random vellocity if False
    env_config['test'] = True
    env_config['worker_index'] = 1 #os.getpid()
    env_config['vx_low'] = 35.0
    env_config['vx_high'] = 65.0
    env_config['res_ux_tol'] =  1.0e-3
    env_config['res_uy_tol'] =  1.0e-3
    env_config['res_p_tol'] =  5.0e-2
    env_config['res_k_tol'] =  1.0e-3
    env_config['res_eps_tol'] =  1.0e-3
    env_config['reward_type'] = 1 # 1: terminal, 2: at each time step
    env_config['states_type'] = 1 # 1: single state, 2: k states history
    dp = dynamic_parameters(env_config)   
    
    done = False
    
    # action parameters for every 200 time steps
#    actions_p = [0.7,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
#    actions_u = [0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
    
    actions_u = [0.8,0.95,0.76,0.2,0.2,0.95,0.6,0.6]
    actions_p = [0.8,0.2,0.2,0.2,0.2,0.2,0.2,0.2]

#    actions_p = [0.2,0.2,0.2,0.9,0.732787489891052,0.2,0.2,0.2,0.9]
#    actions_u = [0.327574849128723,0.509932279586792,0.495954900979996,0.2,0.310916066169739,0.9,0.2,0.40909543633461,0.9]
    k = 0    
    start = time.time()
    state_0 = dp.reset()
    reward = 0
    print(k, ' ', state_0, ' ', reward, ' ', done)
    while not done:
        action = [actions_p[0],actions_u[0]]
        state, reward, done, dict_empty = dp.step(action)
#        if k < 5:
        k = k + 1
        print(k, ' ', state, ' ', reward, ' ', done)
    
    print('CPU Time = ', time.time() - start)
