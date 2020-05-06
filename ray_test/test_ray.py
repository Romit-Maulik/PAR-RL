import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0,HERE)

import argparse
import numpy as np
import ray

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()

ray.init(redis_address=args.ray_address)
print('Nodes:',len(ray.nodes()))
print('Available resources:',ray.available_resources())

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(6)]
print(ray.get(futures))