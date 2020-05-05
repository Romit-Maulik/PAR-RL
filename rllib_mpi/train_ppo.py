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
import ray.rllib.agents.ppo as ppo

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

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
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init(address=args.ray_address)
    ModelCatalog.register_custom_model("my_model", CustomModel)

    config = ppo.DEFAULT_CONFIG.copy()
    #config["log_level"] = "WARN"
    #config["num_gpus"] = 0
    config["num_workers"] = 16
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
    trainer = ppo.PPOTrainer(config=config, env="myenv")

    # Can optionally call trainer.restore(path) to load a checkpoint.
    for i in range(10):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

    # Final save
    checkpoint = trainer.save()
    print("Final checkpoint saved at", checkpoint)

    f = open("rl_checkpoint",'w')
    f.write(checkpoint)
    f.close()
