"""
Created on Tue Jun  9 11:19:50 2020

@author: suraj
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,HERE)

import argparse
import numpy as np
from scipy.integrate import odeint
import gym
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

# Algorithms
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
import ray.rllib.agents.a3c.a3c as a3c
import ray.rllib.agents.a3c.a2c as a2c

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from time import time

from lorenz import lorenzEnv_transient

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()

'''
Custom environment
'''

register_env("myenv", lambda config: lorenzEnv_transient(config))

if __name__ == "__main__":
    #    print(args.ray_address)
    ray.init(redis_address=args.ray_address)
    
#    config = appo.DEFAULT_CONFIG.copy()
#    config = ppo.DEFAULT_CONFIG.copy()
    config = a3c.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = 1 #int(ray.available_resources()['CPU'])
#    config["lr"] = 2.5e-4
    config["horizon"] = 4000
#    config["train_batch_size"] = 4000
#    config["batch_mode"] = "complete_episodes"
#    config["reduce_results"] = False
#    config["vf_clip_param"] = 1000
#    config["sgd_minibatch_size"] = 1  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
#    config["num_sgd_iter"] = 4          # Number of SGD epochs to execute per train batch
#    config["model"]["fcnet_hiddens"] = [64,64]
#    config["model"]["use_lstm"] = True
    
    # Environmental parameters
    env_params = {}
    env_params['n'] = 1
#    env_params['episode'] = 4000
    config["env_config"] = env_params

    # Trainer
#    trainer = appo.APPOTrainer(config=config, env="myenv")
#    trainer = ppo.PPOTrainer(config=config, env="myenv")
#    trainer = a2c.A2CTrainer(config=config, env="myenv")
    trainer = a3c.A3CTrainer(config=config, env="myenv")
    
#    trainer.restore('./APPO_myenv_2020-06-08_12-48-56ir6vtjhs/checkpoint_200/checkpoint-200')
#    trainer.restore('./APPO_myenv_2020-06-08_12-48-56ir6vtjhs/checkpoint_200/checkpoint-200')
#    trainer.restore('./A2C_myenv_2020-06-11_10-11-28xhd8gbys/checkpoint_200/checkpoint-200')
    trainer.restore('./A3C_myenv_2020-06-10_17-24-50owtfp_nk/checkpoint_200/checkpoint-200')
    
    s0 = [1,12,9]

    rho = 20.0
    sigma = 10.0
    beta = 8.0 / 3.0
    
    sigmap = 0.0
    rhop = 0.0
    betap = 0.0
    
    def fuc(state, t):
        x, y, z = state  # Unpack the state vector
        return (sigma) * (y - x), x * (rho - z) - y, x * y - (beta) * z  # Derivatives
    
    def fc(state, t):
        x, y, z = state  # Unpack the state vector
        return (sigma+sigmap) * (y - x), x * (rho+rhop - z) - y, x * y - (beta+betap) * z  # Derivatives
    
    nt = 2500
    tf = 50.0
    dt = tf/nt
    all_states_controlled = np.zeros((nt+1,7))
    all_states_uncontrolled = np.zeros((nt+1,4))
    ts = 0
    spc = s0
    spuc = s0
    
    for k in range(1,nt+1):
        t = np.linspace(ts,ts+dt,2)
        snc = odeint(fc,spc,t)
        snuc = odeint(fuc,spuc,t)
        ts = t[-1]
        xdot, ydot, zdot = fc(snc[-1,:], t)
        obs = np.array([snc[-1,0],snc[-1,1],snc[-1,2],xdot,ydot,zdot])
        action = trainer.compute_action(obs)
        rhop, sigmap, betap = action
        all_states_controlled[k,:] = np.hstack((ts,snc[-1,:],action))
        all_states_uncontrolled[k,:] = np.hstack((ts,snuc[-1,:]))
        spc = np.copy(snc[-1,:])
        spuc = np.copy(snuc[-1,:])
    
    vel_magnitude = np.sqrt(np.sum(all_states_controlled[:,1:]**2,axis=1))
    fig = plt.figure()
    plt.plot(all_states_uncontrolled[:,0],all_states_uncontrolled[:,1]) 
    plt.plot(all_states_controlled[:,0],all_states_controlled[:,1])  
    plt.draw()
    plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(all_states_uncontrolled[:, 1], all_states_uncontrolled[:, 2], all_states_uncontrolled[:, 3],
            label='Unontrolled')
    ax.plot(all_states_controlled[:, 1], all_states_controlled[:, 2], all_states_controlled[:, 3],
            label='Controlled')
    ax.legend()
    plt.draw()
    plt.show()
