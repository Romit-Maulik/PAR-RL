#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:24:58 2020

@author: suraj
"""

import gym
import subprocess

class CustomEnv(gym.Env):
    
    def __init__(self):
        # initialize the environment
        print('Env initialized')
#        subprocess.call(['ls'], shell=True)
        
    def step(self):
        # here we will add code for step method
        subprocess.call(['python ./mesh/generate_msh.py'], shell=True)
        print('Step success')
    
    def reset(self):
        # here we will add code to reset states 
        print('Env reset')
        
