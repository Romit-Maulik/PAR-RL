# Initializing Ray on Theta

The following document has instructions for initializing Ray on Theta. As of now solely Ray version 0.7.6 has been tested successfully. 

## Dependencies
0. Python 3.6.8
1. Ray 0.7.6 - Install with `pip install ray==0.7.6` 

# How does Ray work?

One "starts" Ray on multiple nodes and may communicate between them through their IP addresses. This allows for building and running distributed applications such as hyperparameter optimization, reinforcement learning or distributed supervised learning. We highly recommend some reading of their [walkthrough](https://docs.ray.io/en/latest/walkthrough.html) before proceeding.

A condensed example to explain Ray is shown below. Assume that you have access to multiple nodes that can "talk" to each other through their IP addresses. You may start Ray on one of these nodes and designate this as the "head" as follows:
```
ray start --num_cpus 1 --redis-port=10100
```
this will provide you a `head_redis_address` which is important for communicating among nodes. Note also that we have set an argument `--num_cpus 1` which can be set to a larger number if you can allocate more "ranks" on one node (note the parallel with MPI/OpenMP).

After the head node is initialized, start workers on other nodes by running
```
ray start --num_cpus 1 --address={head_redis_address}
```
where `head_redis_address` was obtained after the head node process was started. The number of "ranks" utilization per node can be increased by changing the `--num_cpus 1` argument. If this is successfully executed - you may now use Ray for several distributed tasks. 

Note that once Ray is started on the head and all workers, a single python script can be executed from the head node and take full advantage of parallelism. For that, your python code that uses Ray should have the following statement
```
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ray-address")
args = parser.parse_args()
ray.init(address=args.ray_address)
```


A single script (lets call this `start_ray.py`) can accomplish this by using `mpi4py` as follows:


```
import os
import sys
import subprocess
from subprocess import Popen, PIPE, CalledProcessError

import socket
import signal
import logging
import psutil
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
        subprocess.run(
            f'ray start --head \
                    --num-cpus 1 \
                    --node-ip-address={head_ip} \
                    --redis-port={REDIS_PORT}',
            shell=True,
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT
        )

def run_ray_worker(head_redis_address):
    with open(f'ray.log.{rank}', 'wb') as fp:
        subprocess.run(
            f'ray start --redis-address={head_redis_address} \
                    --num-cpus 1',
            shell=True,
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT
        )

def fetch_ip():
    # import urllib.request
    # external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    # return external_ip
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

    comm.barrier() # waiting for ray_workers to start

    logging.info('Workers are all running!')

    logging.info('Ready to start driver!')

    return head_redis_address


def worker():
    head_redis_address = None

    logging.info('Waiting for broadcast...')
    head_redis_address = comm.bcast(head_redis_address, root=0)
    logging.info(f'Broadcast done... received head_redis_address= {head_redis_address}')

    comm.barrier()

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
    else: 
        worker()

    comm.barrier()

    if rank == 0:
        # Run the python script to do run something in a distributed fashion
        exec_string = "python my_ray_script.py --ray-address='"+str(head_redis_address)+r"'"
        subprocess.run(exec_string,shell=True,check=True)
        logging.info("My code that uses Ray is invoked successfully. Exiting.")

    comm.barrier()
    print(str(rank)+' rank worker here')

    # Stop all ranks of ray
    ray_stop()
    comm.barrier()
    
    # All finished
    print('Successfully exited')
```

The above script can be be run by using `mpirun -np 4 python start_ray.py` (assuming you have access to 4 compute nodes).

Expanding this capability to Theta is straightforward. First, one needs to construct a virtual environment with the right dependencies. A crucial step is to build a cray compatible `mpi4py` which is available in the companion shell script (`install_mpi4py.sh`). Following this one may execute something along the lines of the following shell script to run Ray on the compute nodes. 

```
#!/bin/bash
#COBALT -t 30
#COBALT -n 2
#COBALT -q debug-cache-quad
#COBALT -A $MY_PROJECT
#COBALT --attrs enable_ssh=1:ssds=required:ssd_size=128

echo "Running Cobalt Job $COBALT_JOBID."

module unload trackdeps
module unload darshan
module unload xalt
# export MPICH_GNI_FORK_MODE=FULLCOPY # otherwise, fork() causes segfaults above 1024 nodes
export PMI_NO_FORK=1 # otherwise, mpi4py-enabled Python apps with custom signal handlers do not respond to sigterm
export KMP_AFFINITY=disabled # this can affect on-node scaling (test this)

# Required for Click_ to work: https://click.palletsprojects.com/en/7.x/python3/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# deactivate core dump
ulimit -c 0

#Loading modules
export PATH=/soft/datascience/anaconda3/bin:$PATH
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH
source activate ray_custom_env

aprun -n $COBALT_JOBSIZE -N 1 --cc none python start_ray.py
```

Reach out to rmaulik@anl.gov, supawar@okstate.edu for more details/troubleshooting.
