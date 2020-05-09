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

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

tf = try_import_tf()


'''
Custom environment
'''


class my_environment(gym.Env):

    def __init__(self, config):
        self.Scalar = config['Scalar']
        print('Scalar value : ', self.Scalar)
        self.observation_space = spaces.MultiDiscrete([ 4, 49, 49, 49, 49 ])
        self.action_space = spaces.Discrete(49)
        self.current_step = 0
        self.intvector = np.asarray([0,0,0,0,0], dtype=np.int64)

    def reset(self):
        self.current_step = 0
        self.intvector = np.asarray([0,0,0,0,0], dtype=np.int64)

        return self.intvector

    def _take_action(self, action):
        self.intvector[self.current_step +1] = action
        self.intvector[0] += 1

    def step(self, action):
         # Need to call a simulation here using a further subprocess
        self._take_action(action)

        self.current_step += 1

        obs = self.intvector
        if self.current_step < 4:
            reward = 0
            done = False
        else:
            self.intvector[1:] += 12
            reward = -(self.Scalar - (self.intvector[3]*self.intvector[2]) / (self.intvector[1]*self.intvector[4]))**2
            self.intvector[1:] -= 12
            done = True

        return obs, reward, done , {}

    def render(self, mode="human", close=False):
        pass


register_env("myenv", lambda config: my_environment(config))

#debugging by following example code 
class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
         name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


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
    ModelCatalog.register_custom_model("my_model", CustomModel)
    register_time = time()

    config = appo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = int(ray.available_resources()['CPU'])
    config["lr"] = 1e-4

    # Add custom model for policy
    model={}
    model["custom_model"] = "my_model"
    config["model"] = model

    # Environmental parameters
    env_params = {}
    env_params['Scalar'] = 1.0/6.931
    config["env_config"] = env_params

    # Trainer
    trainer = appo.APPOTrainer(config=config, env="myenv")
    trainer_time = time()

    # Can optionally call trainer.restore(path) to load a checkpoint.
    with open('Training_iterations.txt','wb',0) as f:
        for i in range(10):
            # Perform one iteration of training the policy with PPO
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

