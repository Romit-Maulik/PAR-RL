from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,HERE)

import argparse
import numpy as np
import math
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

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()

'''
Custom environment
'''

register_env("myenv", lambda config: lorenzEnv_transient(config))


if __name__ == "__main__":
    zero_time = time()
#    print(args.ray_address)
    ray.init(redis_address=args.ray_address)

    with open('Resources.txt','w') as f:
        f.write('Nodes used: '+str(len(ray.nodes()))+'\n')
        f.write('Available resources:'+'\n'),
        f.write(str(ray.available_resources())+'\n')
        f.flush()
        os.fsync(f)
    f.close()

    connect_time = time()
    register_time = time()

    config = appo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = 1 #int(ray.available_resources()['CPU'])
    config["lr"] = 2.5e-4
    config["horizon"] = 4000
    config["sample_batch_size"] = 1000
    config["train_batch_size"] = 4000
    config['metrics_smoothing_episodes'] = 5
    
    # Environmental parameters
    env_params = {}
    env_params['n'] = 1
    config["env_config"] = env_params
    
    total_timesteps = 200000
    # Trainer
    trainer = appo.APPOTrainer(config=config, env="myenv")
    trainer_time = time()
    
    file_results = 'Training_iterations_appo.txt'
    
    # Can optionally call trainer.restore(path) to load a checkpoint.
    result = {'episode_reward_mean':np.float64('nan'), 'timesteps_total':0}
    
    with open(file_results,'wb',0) as f:
#        for i in range(ncount):
        i = 0
        while result['timesteps_total'] <= total_timesteps: 
            # Perform one iteration of training the policy with APPO
            o_string = 'Performing iteration: '+str(i)+'\n'
            o_string = o_string.encode('utf-8')
            f.write(o_string)
            f.flush()
            os.fsync(f)

            init_time = time()
            result = trainer.train()
            o_string = ('Iteration time: '+str(time()-init_time)+'\n').encode('utf-8')
            f.write(o_string)
            f.flush()
            os.fsync(f)
            
            epoch_info = (str(pretty_print(result))+'\n').encode('utf-8')

            f.write(epoch_info)
            f.flush()
            os.fsync(f)            
                
            i = i + 1
            if i%10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)

    f.close()

    iterations_time = time()

    # Final save
    init_time = time()
    checkpoint = trainer.save()
    print("Final checkpoint saved at", checkpoint)

    f = open("rl_checkpoint",'w')
    f.write(checkpoint)
    f.close()

    final_time = time()

    with open('Compute_breakdown.txt','w') as f:
        print('Breakdown of times in this experiment',file=f)
        print('Time to connect:',connect_time-zero_time,file=f)
        print('Time to register environment:',register_time - connect_time,file=f)
        print('Time to setup PPO trainer:',trainer_time - connect_time,file=f)
        print('Time for total iterations:',iterations_time - trainer_time,file=f)
        print('Time to save checkpoint:',final_time - init_time,file=f)
        print('Total time to solution:',final_time - zero_time,file=f)
    f.close()
    
