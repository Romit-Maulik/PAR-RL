#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:30:42 2020

@author: suraj
"""
import subprocess
from subprocess import Popen, PIPE

process = Popen(['touch', 'output1.txt'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
print(stdout)

subprocess.run("sed -i 's/XYZ/ABC/g' output1.txt", shell=True) # change everywhere in whole file

subprocess.run("sed -i '5s/ABC/XYZ/g' output1.txt", shell=True) # change on specific line
