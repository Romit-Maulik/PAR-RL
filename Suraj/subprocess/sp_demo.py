#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:41:16 2020

@author: suraj
"""

import subprocess

#subprocess.run('ls -la', shell=True)
#p1 = subprocess.run(['ls', '-la'])
#
#print(p1.args)
#print(p1.returncode) # 0 - sucessfuly run

#print(p1.stdout) # 

p2 = subprocess.run(['ls', '-la', 'dne'], stdout=subprocess.PIPE)

print(p2.stdout)

print(p2.returncode) # 2 - error

with open('output.txt', 'w') as f:
    p2 = subprocess.run(['ls', '-la'], stdout=f)
    

#%%
p1 = subprocess.run(['cat', 'output.txt'], stdout=subprocess.PIPE, shell=True)

print(p1.stdout)

#p2 = subprocess.run(['grep', '-n', 'test'], capt)