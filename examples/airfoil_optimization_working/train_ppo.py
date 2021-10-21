from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,HERE)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

# Trainign algorithms
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

from shape_environment import shape_optimization
import RL_Inputs as inp
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()

#registering the custom environment
register_env("myenv", lambda config: shape_optimization(config))

if __name__ == "__main__":
    zero_time = time()
    info = ray.init(redis_address=args.ray_address)
    print(info)

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

    if inp.trainer == 'PPO':
        config = ppo.DEFAULT_CONFIG.copy()
    elif inp.trainer == 'APPO':
        config = appo.DEFAULT_CONFIG.copy()
    elif inp.trainer == 'A3C' :
        config = a3c.DEFAULT_CONFIG.copy()
    else:
        raise ValueError('Invalid trainer selection')

    config["log_level"]   = "WARN"
    config["num_gpus"]    = inp.num_gpus
    config["num_workers"] = inp.num_workers  #int(ray.available_resources()['CPU'])

    config["gamma"]       = 1.0 # discount factor = 1 : episodic tasks

#    config["lr"] = 2.5e-4
#    config["horizon"] = 4000
    config["sgd_minibatch_size"] = inp.sgd_minibatch_size  # Total SGD batch size across all devices for SGD. This defines the minibatch size of each SGD epoch
    config["sample_batch_size"]  = inp.sample_batch_size
    config["train_batch_size"]   = inp.train_batch_size
#    config["min_iter_time_s"] = 10
#    config["batch_mode"] = "complete_episodes"
#    config["reduce_results"] = False
    config["vf_clip_param"]              = inp.vf_clip_param
    config['metrics_smoothing_episodes'] = inp.metrics_smoothing_episodes
#    config["num_sgd_iter"] = 4          # Number of SGD epochs to execute per train batch
#    config["model"]["fcnet_hiddens"] = [64,64]
#    config["model"]["use_lstm"] = True


    # Environmental parameters (gets passed while registering the environment)
    env_params = {}
    env_params['write_interval']    = 100 # write interval for states to be computed
    env_params['end_time']          = 500 # maximum number of time steps
    env_params['end_time_model_1']  = 500

    env_params['test']              = inp.testFlag
    env_params['worker_index']      = 1 #os.getpid()
    env_params['res_ux_tol']        = 1.0e-4
    env_params['res_uy_tol']        = 1.0e-4
    env_params['res_p_tol']         = 1.0e-4
    env_params['res_nutilda_tol']   = 1.0e-4
    env_params['reward_type']       = inp.reward_type
    env_params['states_type']       = inp.states_type
    env_params['uinf_type']         = inp.uinf_type
    env_params['uinf_mean']         = inp.uinf_mean
    env_params['uinf_std']          = inp.uinf_std


    #MultiFidelity Framwork parameters
    env_params['numModels']          = inp.numModels
    env_params['models']             = inp.models
    env_params['modelSwitch']        = inp.modelSwitch
    env_params['framework']          = inp.framework
    env_params['modelMaxIters']      = inp.modelMaxIters

    env_params['statWindowSize']     = inp.statWindowSize
    env_params['convWindowSize']     = inp.convWindowSize
    env_params['convCriterion']      = inp.convCriterion
    env_params['varianceRedPercent'] = inp.varianceRedPercent

    data = np.loadtxt('control_points_range.csv',delimiter=',',skiprows=1, usecols=range(1,4))
    env_params['controlparams_low'] = data[:,1] #Airfoil co-ordinate lower limit
    env_params['controlparams_high']= data[:,2]#Airfoil co-ordinate upper limit

    print('---------------------------')
    print('## Learning Initialization ##')
    print('Loaded from checkpoint : {}'.format(inp.checkpointPath)) if inp.checkpointPath != None else print('No checkpoint loaded')
    print('Num.Episodes : {}'.format(inp.numTrainEpisodes if inp.trainFlag == True else inp.numTestEpisodes))
    print('Framework : {}'.format(inp.framework)) if inp.modelSwitch else print('Framework : Single Model')

    print('Model : ')
    print_string = ['Potential Flow' if inp.models[ii] == 1 else 'RANS' for ii in range(len(inp.uinf_type))]
    print(print_string)

    print('Free stream conditions : ')
    print_string = ['Stochastic' if inp.uinf_type[ii] == 1 else 'Deterministic' for ii in range(len(inp.uinf_type))]
    print(print_string)

    print('Free stream velocity (state) mean :')
    print_string = [inp.uinf_mean[ii] for ii in range(len(inp.uinf_type))]
    print(print_string)

    print('Free stream velocity (state) standard deviation : ')
    print_string = [inp.uinf_std[ii] if inp.uinf_type[ii] == 1 else 'Not applicable' for ii in range(len(inp.uinf_type))]
    print(print_string)
    print('---------------------------')

    config["env_config"] = env_params

    if inp.trainFlag:
        #training the agent in a 'train' environment

        if inp.trainer == 'PPO':
            train_agent = ppo.PPOTrainer(config=config, env="myenv") #specify the agent
        else:
            raise ValueError('Trainer other thatn PPO not yet configured')

        trainer_time = time()
        if inp.checkpointPath != None: train_agent.restore(inp.checkpointPath)  #loading checkpoint
        training_results = os.path.join(inp.savePath, 'Training_results.txt')
        result = {'episodes_total':0}
        results = []
        file_checkpoint = open('./Case_Data/rl_checkpoint_data', 'w')
        file_checkpoint.close()
        with open(training_results,'wb',0) as f:
            i = 0
            while result['episodes_total'] <= inp.numTrainEpisodes:
                # Perform one iteration
                o_string = 'Performing iteration: '+str(i)+'\n'
                o_string = o_string.encode('utf-8')
                f.write(o_string)
                f.flush()
                os.fsync(f)

                init_time = time()
                result    = train_agent.train()
                results.append(result)
                o_string  = ('Iteration time: '+str(time()-init_time)+'\n').encode('utf-8')
                f.write(o_string)
                f.flush()
                os.fsync(f)

                epoch_info = (str(pretty_print(result))+'\n').encode('utf-8')

                f.write(epoch_info)
                f.flush()
                os.fsync(f)
                i = i + 1
                if result['training_iteration'] % 1 == 0:
                    if inp.saveInterCheckPts:
                        file_checkpoint = open('./Case_Data/rl_checkpoint_data', 'a')
                        checkpoint = train_agent.save()
                        print("checkpoint saved at", checkpoint)
                        file_checkpoint.write(checkpoint+'\n')
                        file_checkpoint.close()

                    print('Total episodes : {0} | Episode mean reward : {1:.6f}'\
                    .format(result['episodes_total'], result['episode_reward_mean']))
        f.close()
        results.append(result)
        #episode_rewards = results[-1]['hist_stats']['episode_reward']
        iterations_time = time()

        # Final save
        init_time = time()
        checkpoint = train_agent.save()
        print("Final checkpoint saved at", checkpoint)


        file_checkpoint = open('./Case_Data/rl_checkpoint_data', 'a')
        file_checkpoint.write(checkpoint)
        file_checkpoint.close()

        final_time = time()

        break_down_file = open(os.path.join(inp.savePath, 'Time_elapsed_data.txt'), 'w')
        break_down_file.write('Breakdowon of times in the experiment\n')
        break_down_file.write('Time to connect : {}\n'.format(connect_time-zero_time))
        break_down_file.write('Time to register environment : {}\n'.format(register_time - connect_time))
        break_down_file.write('Time to setup PPO trainer : {}\n'.format(trainer_time - connect_time))
        break_down_file.write('Time for total iterations : {}\n'.format(iterations_time - trainer_time))
        break_down_file.write('Time to save checkpoint : {}\n'.format(final_time - init_time))
        break_down_file.write('Total time to solution: {}\n'.format(final_time - zero_time))
        break_down_file.close()

    if inp.testFlag:
        #execute the agent in the test environment
        #usign the trained policy, execute the actions and observe the reward

        #using pre-config
        trained_config = config.copy()
        if inp.trainer == 'PPO':
            test_agent     = ppo.PPOTrainer(config=trained_config, env="myenv") #specify the test agent
        else:
            raise ValueError('Tester other thatn PPO not jet configured')

        env = shape_optimization(env_params) #environment object
        test_time = time()

        #restoring the agent
        if inp.checkpointPath != None and os.pathexists(inp.checkpointPath):
            test_agent.restore(inp.checkpointPath)
        else:
            raise ValueError('Enter a valid restoring checkpoint path')

        test_results = 'test_results.txt'
        result = {'episodes_total':0}
        results = []

        #executing in the test environment
        file_reward = open('Testing_reward_history.txt', 'w')
        file_reward.write('Variables = reward\n')
        for nEpisode in range (inp.numTestEpisodes):

            done = False
            cumReward = 0

            #resetting the environment
            state = env.reset()
            while not done:
                action                  = test_agent.compute_action(state)
                state, reward, done, _  = env.step(action)
                cumReward += reward

            print('Cummulative reward (Testing phase) for episode : {} is  : {}'.format(nEpisode, cumReward))
            file_reward.write('{} \n'.format(cumReward))
        file_reward.close()


    ray.shutdown()
