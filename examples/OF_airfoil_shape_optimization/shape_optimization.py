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
import pandas as pd
from openpyxl import load_workbook
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

from parametric_airfoil import *

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

mesh = 'blockMesh'
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
class shape_optimization(gym.Env):
    
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
        
        self.vx = env_config['vx']
        self.control_points_low = env_config['controlparams_low']
        self.control_points_high = env_config['controlparams_high']
        
        self.res_ux_tol = env_config['res_ux_tol']
        self.res_uy_tol = env_config['res_uy_tol']
        self.res_p_tol = env_config['res_p_tol']
        self.res_nutilda_tol = env_config['res_nutilda_tol']
        
        self.reward_type = env_config['reward_type']
        self.states_type = env_config['states_type']
        
        num_panels = 50
        num_pts = num_panels + 1
        
        x_rad = np.linspace(0, np.pi, num_pts)
        self.x_cos = (np.cos(x_rad) / 2) + 0.5
        self.x_cos = self.x_cos[1:]
        self.num_cp = 3
        
        self.casename = 'baseCase_'+str(self.worker_index)
        self.csvfile = 'progress_'+str(self.worker_index)+'.csv'  

        self.state = None
        
        #Define the bounded action space
        self.action_space = spaces.Box(self.control_points_low, self.control_points_high, dtype=np.float64)
        
        # Define the unbounded state-space
        if self.states_type == 1:
            high1 = np.array([np.finfo(np.float64).max])
        elif self.states_type == 2:
            high1 = np.array([np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max])
    
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
        
        with open(f'log.convert', 'a') as fp:
            subprocess.run(
                f'cp blockMeshDictGenerator.xlsx ./{self.casename}/blockMeshDictGenerator.xlsx',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
                ) 
            
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
        
        if self.states_type == 1:
            self.state = np.array([self.vx]) 
        elif self.states_type == 2:
            None
                    
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
        
        action_cp = action
        
        airfoil_shape = np.array(bezier_airfoil(self.x_cos, munge_ctlpts(action_cp, self.num_cp, self.num_cp)))
        
        wb = load_workbook(f'./{self.casename}/blockMeshDictGenerator.xlsx')
        ws = wb['Input'] 
        shift = 9
        
        for index, data in enumerate(airfoil_shape):
            ws.cell(row=index+shift, column=1).value = data[0]
            ws.cell(row=index+shift, column=2).value = data[1]
        
        wb.save(f'./{self.casename}/blockMeshDictGenerator.xls')
        
        with open(f'log.convert', 'a') as fp:
            subprocess.run(
                    f'libreoffice --headless --convert-to xlsx ./{self.casename}/*.xls --outdir ./{self.casename}',
                    shell=True,
                    stdout=fp,
                    stderr=subprocess.STDOUT
                    )    
        
        wb = load_workbook(f'./{self.casename}/blockMeshDictGenerator.xlsx',data_only=True)
        sh = wb['blockMesh'] 
        
        with open(f'./{self.casename}/blockMeshDictGenerator.csv', 'w') as f:  
            c = csv.writer(f)
            for r in sh.rows:
                c.writerow([cell.value for cell in r])
                
        pd.set_option("display.max_colwidth", 100)
        df = pd.read_csv(f'./{self.casename}/blockMeshDictGenerator.csv',header=None)
        df = df.fillna('')
        
        # quick fix
        for i in [8,11,12]:
            for j in [47,82,102,142]:
                df[i][j] = str(int(df[i][j]))
                    
        base_filename = f'./{self.casename}/system/blockMeshDict'

        df.to_csv(base_filename, header=None, index=None, quoting=csv.QUOTE_NONE, sep=' ', escapechar = ' ')
        
        now = strftime("%m.%d.%Y-%H.%M.%S", gmtime())
        meshLogFile= f'log.{mesh}-{now}'
        
        # to run on theta       
        proc = subprocess.Popen([f'$FOAM_APPBIN/{mesh} {solveroptions} {self.casename} >> {self.casename}/{meshLogFile}'],
                        shell=True,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        
        proc.wait()
        (stdout, stderr) = proc.communicate()
        
        if proc.returncode != 0:
            logger_m.info(f'{action_cp[0]} mesh failed on {pid}')
        else:
            # mesh successful, run the simulation
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
            res_ux,res_uy,res_p,res_nutilda = 1.0, 1.0, 1.0, 1.0
            logger_m.info(f'{action_cp[0]} coefficient solution diverged in the first step on {pid}')
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
                    res_ux,res_uy,res_p,res_nutilda = 1.0, 1.0, 1.0, 1.0
                    logger_m.info(f'{action_cp[0]} coefficient solution no residual history')
                else:
                    log_history = np.reshape(log_history,[-1,5])
                    itercount = int(log_history[-1,0])
                    res_ux = log_history[-1,1]
                    res_uy = log_history[-1,2]
                    res_p = log_history[-1,3]
                    res_nutilda = log_history[-1,4]


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
            res_ux,res_uy,res_p,res_nutilda = 1.0, 1.0, 1.0, 1.0
            logger_m.info(f'{action_cp[0]} coefficient solution diverged with large magnitude of velocity on {pid}')
            
        if (res_ux < self.res_ux_tol and res_uy < self.res_uy_tol and res_p < self.res_p_tol and \
            res_nutilda < self.res_nutilda_tol ) or itercount == self.end_time:    
            
            logger_m.info(f'{action_cp[0]} coefficient solution converged in {itercount} iterations')
            
            done = True
            force_coeffs = np.loadtxt(f'./{self.casename}/postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat')
            cl = np.average(force_coeffs[-100:,3])
            cd = np.average(force_coeffs[-100:,2])
                        
            if self.reward_type == 1:
                reward = -cl/cd
            elif self.reward_type == 2:
                reward = -cd
                
        if self.states_type == 1:
            self.state = np.array([self.vx]) 
        elif self.states_type == 2:
            None
                
        row = np.array([self.worker_index,self.vx,res_ux,res_uy,res_p,
                        res_nutilda,itercount,reward])
    
        row = np.concatenate((row,action_cp))
        
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
    env_config['write_interval'] = 100 # write interval for states to be computed 
    env_config['end_time'] = 500 # maximum number of time steps
    env_config['test'] = True
    env_config['worker_index'] = 1 #os.getpid()
    env_config['res_ux_tol'] = 1.0e-4
    env_config['res_uy_tol'] = 1.0e-4
    env_config['res_p_tol'] = 1.0e-4
    env_config['res_nutilda_tol'] = 1.0e-4
    env_config['reward_type'] = 1 # 1: terminal, 2: at each time step
    env_config['states_type'] = 1 # 1: single state, 2: k states history
    env_config['vx'] = 25.75
    data = np.loadtxt('control_points_range.csv',delimiter=',',skiprows=1, usecols=range(1,4))
    env_config['controlparams_low'] = data[:,1]
    env_config['controlparams_high'] = data[:,2]
    dp = shape_optimization(env_config)   
    
    done = False

    actions_a1 = np.loadtxt('naca0012_cp_optimized.csv',delimiter=',')

    k = 0    
    start = time.time()
    state_0 = dp.reset()
    reward = 0
    print(k, ' ', state_0, ' ', reward, ' ', done)
    for i in range(1):
        while not done:
            state_0 = dp.reset()
            action = actions_a1.tolist()
            state, reward, done, dict_empty = dp.step(action)
            k = k + 1
            print(k, ' ', state, ' ', reward, ' ', done)
        
        print('CPU Time = ', time.time() - start)
        done = False