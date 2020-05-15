#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:26:41 2020

@author: suraj
"""

import subprocess 

import gym
import envs

# create the custm environment
env = gym.make('CustomEnv-v0')

# 
env.step()
#env.reset()

#subprocess.call(['python generate_msh.py'], shell=True)
