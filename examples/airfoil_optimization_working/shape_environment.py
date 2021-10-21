
"""
Created on Tue Jun  2 13:36:55 2020

@author : Suraj
@co-author : Sahil Bhola

Custom environment to optimize the shape of an airfoil using OpenFoam CFD solver
With Multi-Fidelity evaluation support
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
import pdb

import sys
import subprocess
import shutil
from subprocess import Popen, PIPE, CalledProcessError
import RL_Inputs as inp

# global variables related to OpenFoam
mesh = 'blockMesh'              # command in OpenFoam to perform the meshing
solver = 'simpleFoam'           # command in OpenFoam to solve NS equations
solveroptions = '-case'         # solver options
origcase_1 = 'baseCase_potential'           # directory path of the base case file (For model 1)
origcase = 'baseCase'           # directory path of the base case file



class shape_optimization(gym.Env):
    """ Custom environment for airfoil shape optimization """
    metadata = {'render.modes': ['human'],'video.frames_per_second' : 30}

    def __init__(self, env_config):

        # Remove redundant log files
        with open(f'{origcase}/logr.remove', 'wb') as fp:
            subprocess.run(
                f'rm {origcase}/log.* {origcase}/*.txt',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )

        #Removing stale solutions
        with open(f'{origcase}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {origcase}/0.* {origcase}/[1-9]*',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )

        #Removing stale VTK data
        with open(f'{origcase}/logr.remove', 'a') as fp:
            subprocess.run(
                f'rm -r {origcase}/VTK',
                shell=True,
                stdout=fp,
                stderr=subprocess.STDOUT
            )


        if env_config['test']:
            # for testing worker_index = 1
            self.worker_index = env_config['worker_index']
        else:
            # when actually running the RL algorithm for distributed training, the processor ID (PID)
            # is assiggned as the worker_index
            # in RlLib the PID can be accessed by worker_index of env_config
            self.worker_index = env_config.worker_index

        self.numModels              =   env_config['numModels']
        self.modelList              =   env_config['models']

        self.statWindowSizeList     =   env_config['statWindowSize']
        self.convWindowSizeList     =   env_config['convWindowSize']
        self.convCriterionList      =   env_config['convCriterion']
        self.modelMaxItersList      =   env_config['modelMaxIters']
        self.varianceRedPercentList =   env_config['varianceRedPercent']

        self.modelSwitch            =   env_config['modelSwitch']
        self.framework              =   env_config['framework']



        assert(len(self.modelList) == self.numModels), 'Model list is inconsistent with the number of models'


        if len(self.modelMaxItersList) != self.numModels:
            print('Applying model max iterations :  {} to all models '.format(self.modelMaxItersList[0]))
            time.sleep(0.1)
            self.modelMaxItersList = self.modelMaxItersList*self.numModels

        if len(self.statWindowSizeList) != self.numModels:
            print('Applying Statistics window size :  {} to all models '.format(self.statWindowSizeList[0]))
            time.sleep(0.1)
            self.statWindowSizeList = self.statWindowSizeList*self.numModels

        if len(self.convWindowSizeList) != self.numModels:
            print('Applying Convergence window size :  {} to all models '.format(self.convWindowSizeList[0]))
            time.sleep(0.1)
            self.convWindowSizeList = self.convWindowSizeList*self.numModels

        if len(self.convCriterionList) != self.numModels:
            print('Applying Convergence criterion:  {} to all models '.format(self.convCriterionList[0]))
            time.sleep(0.1)
            self.convCriterionList = self.convCriterionList*self.numModels

        if len(self.varianceRedPercentList) != self.numModels:
            print('Applying variance reduction percentage :  {} to all models '.format(self.varianceRedPercentList[0]))
            time.sleep(0.1)
            self.varianceRedPercentList = self.varianceRedPercentList*self.numModels


        self.globalIterList        = np.zeros((self.numModels)) #Global execution
        self.postSwitchIterList    = np.zeros((self.numModels)) #Relative execution (after switch)
        self.convFlagList          = [False]*self.numModels #convergence flag
        self.switchFlagList        = [False]*self.numModels #switching bewteen models flag

        self.modelFlag         =  0 #Initializing with the first available model

        self.assignModelParameters() #Initialize the model

        self.rewardVarianceMax  = [np.finfo(np.float64).min]*sum(self.convCriterionList)
        self.convVarianceMax    = [np.finfo(np.float64).min]*sum(self.convCriterionList)

        self.statWindow         = np.zeros((sum(self.convCriterionList), self.statWindowSize))
        self.convWindow         = np.zeros((sum(self.convCriterionList), self.convWindowSize))


        # Model 1 parameters (Potential Flow - XFOIL)
        self.end_time_model_1 = env_config['end_time_model_1']

        # Model 2 parameters (Spalart-Almaras)
        self.end_time = env_config['end_time']
        self.write_interval = env_config['write_interval']

        self.uinf = None
        self.uinf_type = env_config['uinf_type']
        self.uinf_mean = env_config['uinf_mean']
        self.uinf_std  = env_config['uinf_std']

        if self.modelSwitch : assert (self.numModels > 1), 'Specify Models to be switched to '

        assert(len(self.uinf_type) == self.numModels), 'specify state input type for each model'
        assert(len(self.uinf_mean) == self.numModels), 'specify state mean for each model'
        assert(len(self.uinf_std) == self.numModels), 'specify state standard deviation for each model'

        self.control_points_low = env_config['controlparams_low']
        self.control_points_high = env_config['controlparams_high']

        self.res_ux_tol = env_config['res_ux_tol']
        self.res_uy_tol = env_config['res_uy_tol']
        self.res_p_tol = env_config['res_p_tol']
        self.res_nutilda_tol = env_config['res_nutilda_tol']

        self.reward_type = env_config['reward_type']
        self.states_type = env_config['states_type']

        # Geometry parameters
        num_panels = 50
        num_pts = num_panels + 1

        x_rad = np.linspace(0, np.pi, num_pts)
        self.x_cos = (np.cos(x_rad) / 2) + 0.5
        self.x_cos = self.x_cos[1:]
        self.num_cp = 3

        self.casename = 'baseCase_'+str(self.worker_index)
        self.csvfile = 'progress_'+str(self.worker_index)+'.csv'

        self.state = None

        # Action and state space w/ additional parameters for the RL algorithm
        self.action_space = spaces.Box(self.control_points_low, self.control_points_high, dtype=np.float64)

        # Define the unbounded state-space
        if self.states_type == 1:
            high1 = np.array([np.finfo(np.float64).max])
        elif self.states_type == 2:
            high1 = np.array([np.finfo(np.float64).max,np.finfo(np.float64).max,np.finfo(np.float64).max,
                          np.finfo(np.float64).max,np.finfo(np.float64).max])

        self.observation_space = spaces.Box(-high1, high1, dtype=np.float64) #State space (unbounded for airfoil geometry)


        self.high_penalty = 10  # penalty for divergence (a hyperprameter)

        self.seed() #Enables reproducibility

        #creating data files
        self.savePath = inp.savePath

        if os.path.exists(self.savePath): shutil.rmtree(self.savePath)
        os.makedirs(os.path.join(self.savePath, 'shape_evolution'))
        shutil.copy('./RL_Inputs.py', os.path.join(self.savePath, 'RL_Inputs.txt'))

        file_reward = open(os.path.join(self.savePath, 'reward_history.txt'), 'w')
        file_reward.write('Variables = reward, variance_ratio\n')
        file_reward.close()

        file_action = open(os.path.join(self.savePath, 'action_history.txt'), 'w')
        file_action.close()


    def seed(self, seed=None):
        """ Seed for random number generator """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def modelSelector(self):
        """
        Select the model that needs to executed
        """
        #Multiple model / state distribution case
        if self.framework == 'TL':
            #Different models / state distribution
            self.modelFlag +=1 #upshift
            self.modelSwitch = False #No more model switch is required
            self.assignModelParameters() #updating the model parameters

            print('Model switch is working')
            pdb.set_trace()

        elif self.framework == 'ML':
            raise ValueError('Multi-Fidelity Framework under construction')

        else:
            raise ValueError('Invalid framework selection for model switch')


    def stateSelector(self):
        """
        Based on the 'modelSelector', the inlet state is selected
        """
        mean = self.uinf_mean[self.modelFlag]

        if self.uinf_type[self.modelFlag] == 1:
            #stochastic
            std  = self.uinf_std[self.modelFlag]
            self.uinf = mean + std*np.random.randn(1)

        elif self.uinf_type[self.modelFlag] == 0:
            #deterministic
            self.uinf = np.array([mean])

        else:
            raise ValueError('Invalid selection of uinf_type')

    def assignModelParameters(self):
        """
        Assign the model specific parameters given a modelFlag
        """
        self.modelID            = self.modelList[self.modelFlag]
        self.modelMaxIters      = self.modelMaxItersList[self.modelFlag]
        self.statWindowSize     = self.statWindowSizeList[self.modelFlag]
        self.convWindowSize     = self.convWindowSizeList[self.modelFlag]
        self.convCriterion      = self.convCriterionList[self.modelFlag]
        self.varianceRedPercent = self.varianceRedPercentList[self.modelFlag]
        self.globalIter         = self.globalIterList[self.modelFlag]
        self.postSwitchIter     = self.postSwitchIterList[self.modelFlag]
        self.convFlag           = self.convFlagList[self.modelFlag]
        self.switchFlag         = self.switchFlagList[self.modelFlag]

    def reset(self):
        """ Reset the state """
        pid = self.worker_index

        #selecting the environment (can differ in the model and / or the Inlet state)
        #if the current model converged, then check for the model, else, directly sample the state

        if self.modelSwitch and self.convFlag == True:
            #If there's a possibility to make a switch and the current model has
            #converged then select the new model
            self.modelSelector()

        self.stateSelector()

        reynoldsNumber = self.uinf*1/(8.6834e-6) #nu=8.6834E-6;chord length = 1

        #resetting Model (1) - Potential Flow
        if self.modelID == 1:

            self.casename_1 = 'baseCase_potential_' + str(pid)
            if os.path.isdir(self.casename_1): shutil.rmtree(self.casename_1)
            shutil.copytree(origcase_1, self.casename_1)
            lines = open("./"+self.casename_1+'/xfoil_input.in').read().splitlines()
            lines[0] = "LOAD ./"+self.casename_1+"/airfoil_data.dat"
            lines[4] = "Visc   "+str(reynoldsNumber.item())
            lines[6] = "./"+self.casename_1+"/polar_file.txt"
            open("./"+self.casename_1+'/xfoil_input.in', 'w').write('\n'.join(lines))

       #resetitng Model (2) - RANS
        if self.modelID == 2:

            self.casename = 'baseCase_' + str(pid)
            self.csvfile = 'progress_' + str(pid) + '.csv'
            orig = SolutionDirectory(origcase, archive=None, paraviewLink=False)
            case = orig.cloneCase(self.casename)

            control_dict = ParsedParameterFile(path.join(self.casename, "system", "controlDict"))
            control_dict["endTime"] = self.end_time
            control_dict["writeInterval"] = self.write_interval
            control_dict.writeFile()

            shutil.copy('Inputs', self.casename)
            lines = open("./" + self.casename + '/Inputs').read().splitlines()
            lines[10] = "UINF            "+str(self.uinf.item())+";"
            open("./" + self.casename + '/Inputs', 'w').write('\n'.join(lines))


            #Resetting the log files
            with open(f'{self.casename}/logr.remove', 'a') as fp:
                subprocess.run(
                    f'rm {self.casename}/log.*',
                    shell=True,
                    stdout=fp,
                    stderr=subprocess.STDOUT
                    )

            # remove old solution directories generated during the simulation
            with open(f'{self.casename}/logr.remove', 'a') as fp:
                subprocess.run(
                    f'rm -r {self.casename}/0.* {self.casename}/[1-9]*',
                    shell=True,
                    stdout=fp,
                    stderr=subprocess.STDOUT
                )

        # assign the state
        if self.states_type == 1:
            # inlet velocty as the state of the RL algorithm
            self.state = self.uinf
        elif self.states_type == 2:
            # no state
            None

        return np.array(self.state)


    # Update the state of the environment
    def step(self, action):
        """Executing the action on the suitable environment"""

        # check if the action is within lower and upper bound
        assert self.action_space.contains(action) , "%r (%s) invalid" %(action, type(action))

        pid = self.worker_index #os.getpid()
        logger_m = logging.getLogger(__name__)
        logging.basicConfig(filename=f'foamstatus_{pid}.log',
                            format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)

        # episode is not finished
        done = False

        file_reward = open(os.path.join(self.savePath, 'reward_history.txt'), 'a')
        file_action = open(os.path.join(self.savePath, 'action_history.txt'), 'a')
        np.savetxt(file_action, np.array([action]) , comments = '')
        file_action.close()


        # assign action to control points
        action_cp = action

        #Airfoil shape generated from the action (parameterized)
        # GirHub repo code: #https://github.com/nathanrooy/aerodynamic-shape-optimization/blob/master/pyfoil/parametric_airfoil.py
        airfoil_shape = np.array(bezier_airfoil(self.x_cos, munge_ctlpts(action_cp, self.num_cp, self.num_cp)))

        self.globalIter     += 1
        self.postSwitchIter += 1

        iter = self.globalIter


        #Executing the action using Model 1
        if self.modelID == 1:

            np.savetxt("./"+self.casename_1+'/airfoil_data.dat', airfoil_shape)
            np.savetxt(os.path.join(self.savePath, "shape_evolution"+ "/airfoil_data_model_1_"+\
                        str(iter)+"_.dat"), airfoil_shape \
                       ,header="Variables = x,y \n"+"ZONE T = GRID I = "+\
                       str(airfoil_shape.shape[0])+" J = 1\n"+"Solutiontime = "+str(iter), \
                       comments=''
                       )

            if os.path.exists("./"+self.casename_1+"polar_file.txt"):    os.remove(self.casename_1+"polar_file.txt")
            proc = subprocess.Popen("xfoil < "+"./"+self.casename_1+"/xfoil_input.in ", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
            proc.wait()
            (stdout, stderr) = proc.communicate()
            polar_data = np.loadtxt("./"+self.casename_1+"/polar_file.txt", skiprows=12)

            if polar_data.shape[0] == 0:
                itercount = np.nan
                logger_m.info(f'xfoil simulation failed on {pid}')
                done = True
                reward = -self.high_penalty
                res_ux, res_uy, res_p, res_nutilda = 1.0, 1.0, 1.0, 1.0  # TODO:notion of nutilda does not exist in xfoil
            else:
                res_ux, res_uy, res_p, res_nutilda = 0, 0, 0, 0
                itercount = self.end_time_model_1
                cl = polar_data[1]
                cd = polar_data[2]
                if self.reward_type == 1:
                    # lift to drag ratio as the reward
                    reward = cl / cd

                elif self.reward_type == 2:
                    reward = -cd
                done = True

        # Executing the action using Model 2
        if self.modelID == 2:

            np.savetxt("./" + self.casename + '/airfoil_data.dat', airfoil_shape)
            np.savetxt(os.path.join(self.savePath, "./shape_evolution"+ \
                        "/airfoil_data_model_2_Proc_"+str(pid)+'_iteration'+str(iter)+"_.dat"),\
                        airfoil_shape,header="Variables = x,y \n"+"ZONE T = GRID I = "+\
                        str(airfoil_shape.shape[0])+" J = 1\n"+"Solutiontime = "+str(iter),\
                        comments=''
                        )



            proc = subprocess.Popen(f'python Mesh_generator.py -p ./{self.casename}', shell=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            now = strftime("%m.%d.%Y-%H.%M.%S", gmtime())
            meshLogFile = f'log.{mesh}-{now}'

            proc = subprocess.Popen(
                [f'$FOAM_APPBIN/{mesh} {solveroptions} {self.casename} >> {self.casename}/{meshLogFile}'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

            proc.wait()
            (stdout, stderr) = proc.communicate()

            if proc.returncode != 0:
                # if the mesh fails, write the information in logger (for debugging purposes)
                logger_m.info(f'{action_cp[0]} mesh failed on {pid}')
            else:
                # mesh successful, run the simulation
                now = strftime("%m.%d.%Y-%H.%M.%S", gmtime())
                solverLogFile = f'log.{solver}-{now}'

                # to run on theta
                proc = subprocess.Popen(
                    [f'$FOAM_APPBIN/{solver} {solveroptions} {self.casename} >> {self.casename}/{solverLogFile}'],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)

                proc.wait()
                (stdout, stderr) = proc.communicate()

            if proc.returncode != 0:
                # if the solution diverges, assign a negative reward (penalize that action)
                # episode is finished
                done = True
                reward = -self.high_penalty
                res_ux, res_uy, res_p, res_nutilda = 1.0, 1.0, 1.0, 1.0
                itercount = np.nan
                logger_m.info(f'{action_cp[0]} coefficient solution diverged in the first step on {pid}')
            else:
                #extracting the residuals
                with open(f'{self.casename}/logr.foamLog', 'a') as fp:
                    subprocess.run(
                        f'./extract_residual.sh {self.casename}/{solverLogFile} {self.casename} {pid}',
                        shell=True,
                        stdout=fp,
                        stderr=subprocess.STDOUT
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        log_history = np.genfromtxt(f'./{self.casename}/residual_{pid}.txt')

                    # if log_histyory is empty, it means that something broke with either CFD simulation
                    # or the residuals are too large. Penalize that action
                    if log_history.size == 0:
                        done = True
                        reward = -self.high_penalty
                        res_ux, res_uy, res_p, res_nutilda = 1.0, 1.0, 1.0, 1.0
                        logger_m.info(f'{action_cp[0]} coefficient solution no residual history')
                    else:
                        # solution is converged, assign the residuals
                        log_history = np.reshape(log_history, [-1, 5])
                        itercount = int(log_history[-1, 0])
                        res_ux = log_history[-1, 1]
                        res_uy = log_history[-1, 2]
                        res_p = log_history[-1, 3]
                        res_nutilda = log_history[-1, 4]

            # convert the solution in vtk format
            proc = subprocess.Popen(
                [f'$FOAM_APPBIN/foamToVTK {solveroptions} {self.casename} >> {self.casename}/logr.vtkoutput'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

            proc.wait()
            (stdout, stderr) = proc.communicate()

            # solution has blown up, it gets caught while converting the solution to VTK format
            # penalize that action
            if proc.returncode != 0:
                done = True
                reward = -self.high_penalty
                res_ux, res_uy, res_p, res_nutilda = 1.0, 1.0, 1.0, 1.0
                logger_m.info(f'{action_cp[0]} coefficient solution diverged with large magnitude of velocity on {pid}')

            # if everything is successful, compute the lift and drag coefficient
            if (res_ux < self.res_ux_tol and res_uy < self.res_uy_tol and res_p < self.res_p_tol and \
                res_nutilda < self.res_nutilda_tol) or itercount == self.end_time:

                logger_m.info(f'{action_cp[0]} coefficient solution converged in {itercount} iterations')

                done = True
                force_coeffs = np.loadtxt(f'./{self.casename}/postProcessing/forceCoeffs1/0/forceCoeffs.dat')
                cl = np.average(force_coeffs[-100:, 3])
                cd = np.average(force_coeffs[-100:, 2])

                if self.reward_type == 1:
                    # lift to drag ratio as the reward
                    reward = cl / cd
                elif self.reward_type == 2:
                    reward = -cd


        # update the state at the end of an episode
        if self.states_type == 1:
            self.state = self.uinf
        elif self.states_type == 2:
            None

        # write the data for debugging, post-processing, etc.
        row = np.array([self.worker_index,self.uinf,res_ux,res_uy,res_p,
                        res_nutilda,itercount,reward])

        row = np.concatenate((row,action_cp))

        with open(self.csvfile, 'a') as csvfile:
            np.savetxt(csvfile, np.reshape(row,[1,-1]), delimiter=",")

        #Checking the model convergence
        if self.modelSwitch:
            varianceRatio = self.modelConvergenceCheck(reward)
            file_reward.write('{0:.5f} {1:.8f}\n'.format(reward, varianceRatio))
        else:
            file_reward.write('{0:.5f}\n'.format(reward))
        print(reward)
        file_reward.close()

        return np.array(self.state), reward, done, {}

    def modelConvergenceCheck(self, reward):
        """Check the convergence of the model
        Based on the convergence criterion model,  convergence is tested.
        Convergence criterion:
            convCriterion == 1 :
            Reward mean is computed over a look-back window of 'statWindowSize', which
            remves the inherent noise in the data. To reach some sort of stationarity in
            the reward, this smooth mean must flatten out. Variance ratio of this data
            is computed that indicated the model convergence.
            Variance ratio : max variance of the smooth mean / current variance
            Typically used for the low fidelilty models

            convCriterion == 0 :
            If the instantaneous episode was penalized, convergence was not achieved; Else,
            the episode converged. This is typically used for the high fidelity models

        Inputs:
            reward : (1, ) : Instantaneous reward
        Outputs:
            convergence flags are updated
        """

        iter = int(self.globalIter)

        #checking the model convergence
        if self.convCriterion == 0: #Based on reward value
            self.convFlag = not(reward == -self.high_penalty)

        elif self.convCriterion == 1: #Based on the variance window
            modelIdx = sum(self.convCriterionList[:self.modelID]) - 1

            if iter <= self.statWindowSize:
                if reward == -self.high_penalty:
                    if iter == 1:
                        #Cold start (will eventually get ironed out)
                        self.statWindow[modelIdx, iter - 1] = reward
                    else:
                        #Copy last reward value (only for convergence evaluation)
                        self.statWindow[modelIdx, iter - 1] = self.statWindow[modelIdx, iter - 2]
                else:
                    self.statWindow[modelIdx, iter - 1] = reward

                #compute the reward statistics over the look back window
                rewardVariance     = np.var(self.statWindow[modelIdx, :iter])

            else:
                self.statWindow[modelIdx, :-1] = self.statWindow[modelIdx, 1:]
                if reward == -self.high_penalty:
                    self.statWindow[modelIdx, -1] = self.statWindow[modelIdx, -2]
                else:
                    self.statWindow[modelIdx, -1] = reward

                #compute the reward statistics over the look back window
                rewardVariance     = np.var(self.statWindow[modelIdx, :])

            # Computing the variance ratio of the reward Mean

            self.rewardVarianceMax[modelIdx] = max(self.rewardVarianceMax[modelIdx], rewardVariance)

            varianceRatio = 1 if iter == 1 else rewardVariance / self.rewardVarianceMax[modelIdx]
            #TODO: Add andy sort of convergence test! (To check for the stationarity)

            #checking the variance reduciton
            if varianceRatio < (1-self.varianceRedPercent):
                self.convFlag = True

            print('Variance Ratio : {}'.format(varianceRatio))
            return varianceRatio

    #rendering function (For environment visualization)
    def render(self, mode='human'):
        self.n +=0

    # close the training session
    def close(self):
        return 0

def main():
    env_config = {}
    ##environment configuration##

    #Global parameters
    env_config['write_interval']   = 100 # write interval for states to be computed
    env_config['end_time']         = 500 # maximum number of time steps
    env_config['end_time_model_1'] = 500
    env_config['test'] = True
    env_config['worker_index'] = 1 #os.getpid()
    env_config['res_ux_tol'] = 1.0e-4
    env_config['res_uy_tol'] = 1.0e-4
    env_config['res_p_tol'] = 1.0e-4
    env_config['res_nutilda_tol'] = 1.0e-4
    env_config['reward_type'] = inp.reward_type  # 1: cl/cd, 2: cd
    env_config['states_type'] =  inp.states_type # 1: single state, 2: k states history
    env_config['uinf_type']   =  inp.uinf_type # 1: stochastic; 2: deterministic
    env_config['uinf_mean'] = inp.uinf_mean  # Mean of the freestream velocity
    env_config['uinf_std'] = inp.uinf_std    # Standard deviation (for stochastic inflow conditions)
    data = np.loadtxt('control_points_range.csv',delimiter=',',skiprows=1, usecols=range(1,4))
    env_config['controlparams_low'] = data[:,1] #Airfoil co-ordinate lower limit
    env_config['controlparams_high'] = data[:,2]#Airfoil co-ordinate upper limit

    #Multi-Fidelity framework parameters

    env_config['numModels']          = inp.numModels # Number of available models for learning
    env_config['modelID']            = inp.initModelID  # Initial model
    env_config['modelSwitch']        = inp.modelSwitch # False : modelID will used; True: models(s)
    env_config['modelMaxIters']      = inp.modelMaxIters #Switching between the same model (diff. state)
    env_config['modelMaxIters']      = inp.modelMaxIters #Max Episodes after switch
    env_config['statWindowSize']     = inp.statWindowSize # Statistics window
    env_config['convWindowSize']     = inp.convWindowSize # Convergence window
    env_config['convCriterion']      = inp.convCriterion #; 0:Immediate reward; 1: Window
    env_config['varianceRedPercent'] = inp.varianceRedPercent #variance reduction percentage

    env = shape_optimization(env_config)
    done = False

    #reading the action from .csv file
    action_dummy = np.loadtxt('naca0012_cp_optimized.csv',delimiter=',')

    start = time()
    initial_state = env.reset()
    reward = 0

    output = 'State : {0:.5f} reward : {1:.5f} done : {2}'

    print(output.format(initial_state.item(), reward, done))

    for nEpisode in range(1):
        subIter = 0
        print('## #Episode : '+str(nEpisode + 1)+' ##')
        while not done:
            #reset the environment
            initial_state = env.reset()

            #checking the test action
            action = action_dummy.tolist()

            #executing the action on the environment
            state, reward, done, _ = env.step(action)

            subIter +=1
            print(output.format(state.item(), reward, done))

        print('Time elapsed : {} seconds'.format(time() - start))
        done = False

if __name__ == '__main__':
    main()
