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
from ray.rllib.optimizers import AsyncGradientsOptimizer

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


if __name__ == "__main__":
    zero_time = time()
    ray.init(redis_address=args.ray_address)

    with open('Resources.txt','w') as f:
        print('***********************************************************')
        print('Nodes used:',len(ray.nodes()),file=f)
        print('Available resources:',ray.available_resources(),file=f)
        print('***********************************************************')
    f.close()

    connect_time = time()

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = int(ray.available_resources()['CPU'])
    config["lr"] = 1e-4

    # Trainer
    def make_async_optimizer(workers, config):
        return AsyncGradientsOptimizer(workers, grads_per_step=10)

    CustomTrainer = ppo.PPOTrainer.with_updates(
        make_policy_optimizer=make_async_optimizer)

    trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
    trainer_time = time()

    # Can optionally call trainer.restore(path) to load a checkpoint.
    with open('Training_iterations.txt','w+') as f:
        for i in range(10):
            # Perform one iteration of training the policy with PPO
            f.write('Performing iteration: '+str(i)+'\n')
            init_time = time()
            result = trainer.train()
            f.write('Iteration time: '+str(time()-init_time)+'\n')
            f.write(pretty_print(result)+'\n')
            f.flush()
    f.close()

    iterations_time = time()

    # Final save
    init_time = time()
    checkpoint = trainer.save()

    f = open("rl_checkpoint.txt",'w')
    f.write(checkpoint)
    f.close()
    final_time = time()

    with open('Compute_breakdown.txt','w') as f:
        print('Breakdown of times in this experiment',file=f)
        print('Time to connect:',connect_time-zero_time,file=f)
        print('Time to setup PPO trainer:',trainer_time - connect_time,file=f)
        print('Time for total iterations:',iterations_time - trainer_time,file=f)
        print('Time to save checkpoint:',final_time - init_time,file=f)
        print('Total time to solution:',final_time - zero_time,file=f)
    f.close()
