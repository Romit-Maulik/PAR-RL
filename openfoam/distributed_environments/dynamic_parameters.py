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

solver = 'simpleFoam'
solveroptions='-case'
origcase = 'baseCase'

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
        
        self.number_steps = 0
        self.update_frequency = env_config['update_frequency']
        self.pid = env_config['pid']
        self.casename = 'baseCase_'+str(self.pid)
        self.csvfile = 'progress_'+str(self.pid)+'.csv'  
#        self.max_steps = env_config['max_steps']
        self.state = None
        
#        orig = SolutionDirectory(origcase,archive=None,paraviewLink=False)
#        case=orig.cloneCase(self.casename )
#        
#        write_interval = ParsedParameterFile(path.join(self.casename,"system", "controlDict"))
#        write_interval["writeInterval"] = self.update_frequency
#        write_interval.writeFile()
        
        relax_p_low = 0.2
        relax_u_low = 0.2
        relax_p_high = 0.9
        relax_u_high = 0.9

        # Lower bound for relaxation factors
        relaxation_low = np.array([relax_p_low,relax_u_low])        
        
        # Upper bound for relaxation factors
        relaxation_high = np.array([relax_p_high,relax_u_high])
        
        #Define the bounded action space
        self.action_space = spaces.Box(relaxation_low, relaxation_high, dtype=np.float64)
        
        # Define the unbounded state-space
        self.vx = 0.0
        self.vx_low = 4.0
        self.vx_high = 8.0
        self.vx_all = np.array([7.333,8.0,4.0])
        
        velocity_low = np.array([self.vx_low])
        velocity_high = np.array([self.vx_high])

        # Define the unbounded state-space
        high1 = np.array([np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high1, high1, dtype=np.float64)
        
        # penalty for divergence
        self.high_penalty = 5000
        
        self.seed()
            
    # Seed for random number generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset the environment at the end of each episode
    def reset(self):        
        pid = os.getpid()
        self.casename = 'baseCase_'+str(pid)
        self.csvfile = 'progress_'+str(pid)+'.csv'  
        orig = SolutionDirectory(origcase,archive=None,paraviewLink=False)
        case=orig.cloneCase(self.casename )
        
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
                
        # set up the case with random velicity condition between u = 4 to 8 m/s
        self.vx = self.np_random.uniform(low=self.vx_low, high=self.vx_high, size=(1,)) #self.vx_all[0] # additional velicities to be added
        velBC = ParsedParameterFile(path.join(self.casename,"0", "U"))
        velBC["boundaryField"]["inlet"]["value"].setUniform(Vector(self.vx,0,0))
        velBC.writeFile()
        
        # convert solution files to vtk format
        with open(f'{self.casename}/logr.vtkoutput', 'wb') as fp:
            subprocess.run(
                f'foamToVTK {solveroptions} {self.casename}',
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
        
        U2 = np.average(np.square(Ub))*3 + np.average(np.square(Um))*3
        self.state = np.array([U2]) #self.np_random.uniform(low=self.vx_low, high=self.vx_high, size=(1,))
        
        return np.array(self.state)
    
    # Update the state of the environment
    def step(self, action):
        
        assert self.action_space.contains(action) , "%r (%s) invalid" %(action, type(action))
        
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
    #    subprocess.run(
    #            f'$FOAM_APPBIN/{solver} {solveroptions} >> {solverLogFile}',
    #            shell=True,
    #            check=True,
    #            stdout=fp,
    #            stderr=subprocess.STDOUT
    #        )
        
#        subprocess.run(
#                f'{solver} {solveroptions} >> {solverLogFile}',
#                shell=True,
#                check=True,
#                stderr=subprocess.STDOUT
#            )
        
#        proc = subprocess.Popen(['simpleFoam -case baseCase >> log1'],
#                        shell=True,
#                        stdout=subprocess.PIPE, 
#                        stderr=subprocess.PIPE)
    
        proc = subprocess.Popen([f'{solver} {solveroptions} {self.casename} >> {self.casename}/{solverLogFile}'],
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
            res_ux,res_uy,res_p = 1.0, 1.0, 1.0
        else:
            with open(f'{self.casename}/logr.foamLog', 'a') as fp:   
                pid = os.getpid()
                subprocess.run(
                        f'./extract_residual.sh {self.casename}/{solverLogFile} {self.casename} {pid}',
                        shell=True,
                        stdout = fp,
                        stderr=subprocess.STDOUT
                    )
                log_history = np.genfromtxt(f'./{self.casename}/residual_{pid}.txt')
                log_history = np.reshape(log_history,[-1,6])
                itercount = int(log_history[-1,0])
                res_ux = log_history[-1,1]
                res_uy = log_history[-1,2]
                res_p = log_history[-1,3]
                res_k = log_history[-1,4]
                res_eps = log_history[-1,5]
                
#                subprocess.run(
#                        f'foamLog {solverLogFile}',
#                        shell=True,
#                        stdout = fp,
#                        stderr=subprocess.STDOUT
#                    )
#            ux_0 = np.genfromtxt('.baseCase/logs/Ux_0')
#            uy_0 = np.genfromtxt('.baseCase/logs/Uy_0')
#            p_0 = np.genfromtxt('.baseCase/logs/p_0')
#            k_0 = np.genfromtxt('.baseCase/logs/k_0')
#            eps_0 = np.genfromtxt('.baseCase/logs/epsilon_0')
#            res_ux = np.reshape(ux_0,[-1,2])[-1,1]
#            res_uy = np.reshape(uy_0,[-1,2])[-1,1]
#            res_p = np.reshape(p_0,[-1,2])[-1,1]
#            res_k = np.reshape(k_0,[-1,2])[-1,1]
#            res_eps = np.reshape(eps_0,[-1,2])[-1,1]
#            itercount = int(np.reshape(ux_0,[-1,2])[-1,0])
            
            if res_ux < 1e-3 and res_uy < 1e-3 and res_p < 1e-2 and res_k < 1e-3 and res_eps < 1e-3:
                reward = -itercount
                done = True
            else:
                reward = 0
                
#            file = open(f'{solverLogFile}', 'r')
#            lines = file.read().splitlines()
#            line = lines[-9]
#            columns = [col.strip() for col in line.split(' ') if col]
#            check = columns[-1]
#            
#            if check != 'iterations': 
#                line = lines[-15]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_ux = np.float64(columns[7][:-1])
#                line = lines[-14]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_uy = np.float64(columns[7][:-1])
#                line = lines[-13]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_p = np.float64(columns[7][:-1])
#                line = lines[-17]
#                columns = [col.strip() for col in line.split(' ') if col]
#                itercount = np.int(columns[2])
#                reward = 0
#                    
#            else:
#                # solution converged
#                done = True
#                line = lines[-18]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_ux = np.float64(columns[7][:-1])
#                line = lines[-17]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_uy = np.float64(columns[7][:-1])
#                line = lines[-16]
#                columns = [col.strip() for col in line.split(' ') if col]
#                res_p = np.float64(columns[7][:-1])
#                line = lines[-20]
#                columns = [col.strip() for col in line.split(' ') if col]
#                itercount = np.int(columns[2])
#                reward = -itercount
        
        prev_folder = self.number_steps - self.update_frequency
        if prev_folder != 0:
            with open(f'{self.casename}/logr.remove', 'a') as fp:
                subprocess.run(
                    f'rm -r {self.casename}/{prev_folder}',
                    shell=True,
                    stdout=fp,
                    stderr=subprocess.STDOUT
                )
        
        # convert solution files to vtk format
#        with open('logr.vtkoutput', 'a') as fp:
#            subprocess.run(
#                f'foamToVTK',
#                shell=True,
#                stdout=fp,
#                stderr=subprocess.STDOUT
#            )
        
        proc = subprocess.Popen([f"foamToVTK {solveroptions} {self.casename} >> {self.casename}/logr.vtkoutput"],
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
            res_ux,res_uy,res_p = 1.0, 1.0, 1.0
            
        if itercount != self.high_penalty:    
            inlet = vtki.PolyData(f'./{self.casename}/VTK/inlet/inlet_{itercount}.vtk')
            Ub = inlet.cell_arrays['U']
            Ub = np.array(Ub)
            
            mesh = vtki.UnstructuredGrid(f'./{self.casename}/VTK/{self.casename}_{itercount}.vtk')
            Um = mesh.cell_arrays['U']
            Um = np.array(Um)
            
            U2 = np.average(np.square(Ub))*3 + np.average(np.square(Um))*3
        else:
            U2 = self.high_penalty
            
        self.state = np.array([U2])
                
        row = np.array([os.getpid(),self.vx,relax_u,relax_p,itercount,U2,res_ux,res_uy,res_p,reward])
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
    env_config['max_steps'] = 5000 # maximum number of time steps
    env_config['pid'] = os.getpid()
    dp = dynamic_parameters(env_config)   
    
    done = False
    
    # action parameters for every 200 time steps
    actions_p = [0.7,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
    actions_u = [0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
    
#    actions_p = [0.2,0.2,0.2,0.9,0.732787489891052,0.2,0.2,0.2,0.9]
#    actions_u = [0.327574849128723,0.509932279586792,0.495954900979996,0.2,0.310916066169739,0.9,0.2,0.40909543633461,0.9]
    k = 0    
    start = time.time()
    state_0 = dp.reset()
    reward = 0
    print(k, ' ', state_0, ' ', reward, ' ', done)
    while not done:
        action = [actions_p[k],actions_u[k]]
        state, reward, done, dict_empty = dp.step(action)
#        if k < 5:
        k = k + 1
        print(k, ' ', state, ' ', reward, ' ', done)
    
    print('CPU Time = ', time.time() - start)