"""
Created on Mon Jul 20 09:49:49 2020

@author: suraj
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

#%%
pid = 1
casename = 'baseCase_'+str(pid)
orig = SolutionDirectory(origcase,archive=None,paraviewLink=False)
case=orig.cloneCase(casename )

turb_model = ParsedParameterFile(path.join(casename,"constant", "turbulenceProperties"))
turb_model["RAS"]["kOmegaSSTCoeffs"]["a1"] = 0.58
turb_model.writeFile()

#%%
mesh = vtki.UnstructuredGrid(f'./{self.casename}/VTK/{self.casename}_0.vtk')
Um = mesh.cell_arrays['U']
Um = np.array(Um)