#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:20:27 2020

@author: suraj
"""

from gym.envs.registration import register

register(id='CustomEnv-v0',
         entry_point='envs.custom_env_dir:CustomEnv'
         )