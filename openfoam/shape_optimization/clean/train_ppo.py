from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,HERE)

import argparse
import numpy as np
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

#from dynamic_parameters import dynamic_parameters
from shape_optimization import shape_optimization

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()

'''
Custom environment
'''

register_env("myenv", lambda config: shape_optimization(config))


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
    
#    resources = ray.get_resource_ids()  
#    cpus = [v[0] for v in resources['CPU']] 


#    config = appo.DEFAULT_CONFIG.copy()
    config = ppo.DEFAULT_CONFIG.copy()
#    config = a3c.DEFAULT_CONFIG.copy()
#    config = a3c.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = 1  #int(ray.available_resources()['CPU'])
    
    config["gamma"] = 1.0 # discount factor = 1 : episodic tasks
    
#    config["lr"] = 2.5e-4
#    config["horizon"] = 4000
    config["sgd_minibatch_size"] = 8  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
    config["sample_batch_size"] = 20
    config["train_batch_size"] = 20
#    config["min_iter_time_s"] = 10
#    config["batch_mode"] = "complete_episodes"
#    config["reduce_results"] = False
    config["vf_clip_param"] = 10
    config['metrics_smoothing_episodes'] = 10
#    config["num_sgd_iter"] = 4          # Number of SGD epochs to execute per train batch
#    config["model"]["fcnet_hiddens"] = [64,64]
#    config["model"]["use_lstm"] = True
    

    # Environmental parameters
    env_params = {}
    env_params['write_interval'] = 100 # write interval for states to be computed 
    env_params['end_time'] = 500 # maximum number of time steps
    
    env_params['test'] = False
    env_params['worker_index'] = 1 #os.getpid()
    env_params['res_ux_tol'] = 1.0e-4
    env_params['res_uy_tol'] = 1.0e-4
    env_params['res_p_tol'] = 1.0e-4
    env_params['res_nutilda_tol'] = 1.0e-4
    env_params['reward_type'] = 1 # 1: cl/cd, 2: cd
    env_params['states_type'] = 1 # 1: single state, 2: k states history
    env_params['vx'] = 25.75
    data = np.loadtxt('base_naca0012_cp_optimized.csv',delimiter=',',skiprows=1, usecols=range(1,4))
    env_params['controlparams_low'] = data[:,1]
    env_params['controlparams_high'] = data[:,2]
    
    config["env_config"] = env_params

    # Trainer
#    trainer = appo.APPOTrainer(config=config, env="myenv")
    trainer = ppo.PPOTrainer(config=config, env="myenv")
#    trainer = a3c.A3CTrainer(config=config, env="myenv")
#    trainer = a2c.A2CTrainer(config=config, env="myenv")
    trainer_time = time()
    
#    trainer.restore('./PPO_myenv_2020-07-10_18-04-27e0smkyiu/checkpoint_35/checkpoint-35')
    
    file_results = 'Training_iterations_ppo.txt'
    
    # Can optionally call trainer.restore(path) to load a checkpoint.
    result = {'episodes_total':0}
    results = []
    with open(file_results,'wb',0) as f:
#        for i in range(ncount):
        i = 0
        while result['episodes_total'] <= 1000:
            # Perform one iteration of training the policy with APPO
            o_string = 'Performing iteration: '+str(i)+'\n'
            o_string = o_string.encode('utf-8')
            f.write(o_string)
            f.flush()
            os.fsync(f)

            init_time = time()
            result = trainer.train()
            results.append(result)
            o_string = ('Iteration time: '+str(time()-init_time)+'\n').encode('utf-8')
            f.write(o_string)
            f.flush()
            os.fsync(f)
            
            epoch_info = (str(pretty_print(result))+'\n').encode('utf-8')

            f.write(epoch_info)
            f.flush()
            os.fsync(f)
            i = i + 1
            
            if result['training_iteration'] % 1 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)

    f.close()
    results.append(result)
#    episode_rewards = results[-1]['hist_stats']['episode_reward']
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
    
