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

from turb_model_parameters_sdp import turb_model_parameters

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()

'''
Custom environment
'''

register_env("myenv", lambda config: turb_model_parameters(config))


if __name__ == "__main__":
    zero_time = time()
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
    
    config = ppo.DEFAULT_CONFIG.copy()

    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = int(ray.available_resources()['CPU'])
    
    config["gamma"] = 1.0 # discount factor = 1 : episodic tasks
    
#    config["lr"] = 2.5e-4
    config["sgd_minibatch_size"] = 128 # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
    config["sample_batch_size"] = 200
    config["train_batch_size"] = 200
    config["vf_clip_param"] = 10
    config['metrics_smoothing_episodes'] = 20
   

    # Environmental parameters
    env_params = {}
    env_params['write_interval'] = 500 # write interval for states to be computed 
    env_params['end_time'] = 1500 # maximum number of time steps
    env_params['vx'] = 44.2
    env_params['h'] = 0.0127
    env_params['test'] = False
    kosst_params = np.array([0.85,1.0,0.5,0.856,0.075,0.0828,0.09,0.5555,0.44])
    kosst_params_scaled = np.zeros(12)
    reward_base = np.array([-1.4513])
    env_params['num_parameters'] = 9
    env_params['a1'] = np.array([0.31])
    env_params['reward'] = reward_base  
    env_params['actions_low'] = np.array([0.0,0.0,0.0,0.0,0.012,0.012,0.029,0.0,0.0])
    env_params['actions_high'] = np.array([1.0,1.0,1.0,1.0,0.23,0.23,0.20,1.1,1.1])
    env_params['res_ux_tol'] =  5.0e-4
    env_params['res_uy_tol'] =  5.0e-4
    env_params['res_p_tol'] =  5.0e-2
    env_params['res_k_tol'] =  1.0e-3
    env_params['res_eps_tol'] =  1.0e-3
    env_params['reward_type'] = 2 # 1: L2 norm, 2: normalized square
    env_params['states_type'] = 1 # 1: single state, 
    config["env_config"] = env_params

    # Trainer
    trainer = ppo.PPOTrainer(config=config, env="myenv")

    trainer_time = time()
    
#    trainer.restore('./PPO_myenv_2020-07-10_18-04-27e0smkyiu/checkpoint_35/checkpoint-35')
    
    total_episodes = 4000 # total number of episoded to be trained for
    save_freq = 1         # frequency at which to checkpoint
    file_results = 'Training_iterations_ppo.txt'
    
    result = {'episodes_total':0}
    results = []
    with open(file_results,'wb',0) as f:
        i = 0
        while result['episodes_total'] <= total_episodes:
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
            
            if result['training_iteration'] % save_freq == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)

    f.close()
    results.append(result)
    iterations_time = time()

    # Final save
    init_time = time()
    checkpoint = trainer.save()
    print("Final checkpoint saved at", checkpoint)

    f = open("rl_checkpoint",'w')
    f.write(checkpoint)
    f.close()
    i = i + 1

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
    
