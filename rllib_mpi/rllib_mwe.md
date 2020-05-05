# Minimum working example to reproduce Ray error on Theta:

## Introduction to Ray and distributed RL

Based on the template provided [here](https://github.com/ytopt-team/tuster/blob/master/tuster/system/theta/run.py)
I was able to set up a reinforcement learning search using RLLib and Ray. The idea is as follows:

1. Start ray on the head node and retrieve the `head_redis_address` by running
```
ray start --num_cpus 1 --redis-port=10100
```

2. Start workers on other nodes by running
```
ray start --num_cpus 1 --address={head_redis_address}
```
where `head_redis_address` is obtained after the head node process is started. If this is successfully executed - you are good to execute RLLib in a distributed fashion. 

3. Within your python script which executes RLLib you must have a statement
```
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()
ray.init(address=args.ray_address)
```
which lets you call the script (in our following MWE this will be `train_ppo.py`) as `python train_ppo.py --ray-address head_redis_address`. An important point here is that this must be called on the head node alone and the RL workers will be automatically distributed (the beauty of Ray/RLLib). 

4. All this business can be packaged quite effectively using a script as follows:
```
import os
import sys
import subprocess
from subprocess import Popen

import socket
import signal
import logging
import psutil
from pprint import pformat
import ray
import time
from mpi4py import MPI
from ray.services import get_node_ip_address

from redis.exceptions import ConnectionError

# opening ports as suggested in: https://github.com/ray-project/ray/issues/4393
REDIS_PORT          = 10100
REDIS_SHARD_PORTS   = 20200
NODE_MANAGER_PORT   = 30300
OBJECT_MANAGER_PORT = 40400

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# EXIT
def on_exit(signum, stack):
    ray_stop()


def ray_stop():
    with open('stop.out', 'wb') as fp:
        subprocess.run(
            "ray stop",
            shell=True,
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT
        )

signal.signal(signal.SIGINT, on_exit)
signal.signal(signal.SIGTERM, on_exit)

def run_ray_head(head_ip):
    with open('ray.log.head', 'wb') as fp:
        # subprocess.run(
        #     f'ray start --head \
        #             --num-cpus 1 \
        #             --node-ip-address={head_ip} \
        #             --redis-port={REDIS_PORT} \
        #             --redis-shard-ports={REDIS_SHARD_PORTS} \
        #             --node-manager-port={NODE_MANAGER_PORT} \
        #             --object-manager-port={OBJECT_MANAGER_PORT}',
        #     shell=True,
        #     check=True,
        #     stdout=fp,
        #     stderr=subprocess.STDOUT
        # )

        subprocess.run(
            f'ray start --head --num-cpus 1 --redis-port={REDIS_PORT}',
            shell=True,
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT
        )


def run_ray_worker(head_redis_address):
    with open(f'ray.log.{rank}', 'wb') as fp:
        # --node-manager-port={NODE_MANAGER_PORT} \
        # --object-manager-port={OBJECT_MANAGER_PORT}',
        # subprocess.run(
        #     f'ray start --address={head_redis_address} \
        #             --num-cpus 1 \
        #             --node-ip-address={fetch_ip()} \
        #             --node-manager-port={NODE_MANAGER_PORT} \
        #             --object-manager-port={OBJECT_MANAGER_PORT}',
        #     shell=True,
        #     check=True,
        #     stdout=fp,
        #     stderr=subprocess.STDOUT
        # )

        subprocess.run(
            f'ray start --num-cpus 1 --address={head_redis_address}',
            shell=True,
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT
        )


def fetch_ip():
    return socket.gethostbyname(socket.gethostname())


def master():
    head_ip = fetch_ip()
    if head_ip is None:
        raise RuntimeError("could not fetch head_ip")

    logging.info('Ready to run ray head')

    run_ray_head(head_ip)

    head_redis_address = f'{head_ip}:{REDIS_PORT}'

    logging.info(f'Head started at: {head_redis_address}')

    logging.info(f'Ready to broadcast head_redis_address: {head_redis_address}')

    head_redis_address = comm.bcast(head_redis_address, root=0)

    logging.info('Broadcast done...')

    logging.info('Waiting for workers to start...')

    comm.barrier() # waiting for ray_workers to start - this barrier synchronizes with line 120

    logging.info('Workers are all running!')

    logging.info('Ready to start driver!')

    return head_redis_address


def worker():
    head_redis_address = None

    logging.info('Waiting for broadcast...')
    head_redis_address = comm.bcast(head_redis_address, root=0)
    logging.info(f'Broadcast done... received head_redis_address= {head_redis_address}')

    comm.barrier() # This barrier synchronizes with line 120

    logging.info(f"Worker on rank {rank} with ip {fetch_ip()} will connect to head-redis-address={head_redis_address}")
    run_ray_worker(head_redis_address)
    logging.info(f"Worker on rank {rank} with ip {fetch_ip()} is connected!")

if __name__ == "__main__":

    logging.basicConfig(
        filename='start_ray.log',
        format='%(asctime)s | %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)

    if rank == 0: 
        head_redis_address = master()
        
        # with open('head_redis_address.txt','w') as fp:
        #     fp.write(head_redis_address)
        # fp.close()
    
    else: 
        worker()

    comm.barrier()

    if rank == 0:
        # Run the python script to do RL
        exec_string = "python train_ppo.py --ray-address='"+str(head_redis_address)+r"'"
        with open('rllib_log.out', 'wb') as fp:
            subprocess.run(
                        exec_string,
                        shell=True,
                        check=True,
                        stdout=fp,
                        stderr=subprocess.STDOUT,
        )

        logging.info("RL LIB invoked successfully. Exiting.")

    comm.barrier()
    print(str(rank)+' rank worker here')

    # Stop all ranks of ray
    ray_stop()
    comm.barrier()
    
    # All finished
    print('Successfully exited')
```

where `train_ppo.py` is given by 

```
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
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = 3
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
```
5. This distributed RL runs without any trouble at all on my laptop for 4 workers and can be called by running `mpirun -np 4 python start_ray.py`

## Running on Theta compute nodes

When I try to execute the same procedure by running an interactive job on Theta using
```
qsub -A datascience -t 60 -q debug-cache-quad -n 2 -I
```
and using aprun as `aprun -n 32 -N 16 python start_ray.py`. The logs for starting ray (at `start_ray.log`) show success:

```
05/01/2020 07:36:14 PM | Waiting for broadcast...
05/01/2020 07:36:14 PM | Waiting for broadcast...
05/01/2020 07:36:14 PM | Waiting for broadcast...
05/01/2020 07:36:14 PM | Ready to run ray head
05/01/2020 07:36:25 PM | Head started at: 10.128.15.16:6379
05/01/2020 07:36:25 PM | Ready to broadcast head_redis_address: 10.128.15.16:6379
05/01/2020 07:36:25 PM | Broadcast done... received head_redis_address= 10.128.15.16:6379
05/01/2020 07:36:25 PM | Broadcast done... received head_redis_address= 10.128.15.16:6379
05/01/2020 07:36:25 PM | Broadcast done...
05/01/2020 07:36:25 PM | Broadcast done... received head_redis_address= 10.128.15.16:6379
05/01/2020 07:36:25 PM | Waiting for workers to start...
05/01/2020 07:36:25 PM | Worker on rank 3 with ip 10.128.15.16 will connect to head-redis-address=10.128.15.16:6379
05/01/2020 07:36:25 PM | Worker on rank 2 with ip 10.128.15.16 will connect to head-redis-address=10.128.15.16:6379
05/01/2020 07:36:25 PM | Worker on rank 1 with ip 10.128.15.16 will connect to head-redis-address=10.128.15.16:6379
05/01/2020 07:36:32 PM | Worker on rank 3 with ip 10.128.15.16 is connected!
05/01/2020 07:36:32 PM | Worker on rank 2 with ip 10.128.15.16 is connected!
05/01/2020 07:36:33 PM | Worker on rank 1 with ip 10.128.15.16 is connected!
05/01/2020 07:36:33 PM | Workers are all running!
05/01/2020 07:36:33 PM | Ready to start driver!
```

currently we are debugging the `rllib_log.out` logs to see if the MWE runs successfully and scales appropriately
