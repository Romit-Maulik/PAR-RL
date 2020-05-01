import os
import sys
import subprocess
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
RANK = rank = comm.Get_rank()

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

    comm.barrier() # waiting for ray_workers to start

    logging.info('Workers are all running!')

    logging.info('Ready to start driver!')

    return head_redis_address


def worker():
    head_redis_address = None

    logging.info('Waiting for broadcast...')
    head_redis_address = comm.bcast(head_redis_address, root=0)
    logging.info(f'Broadcast done... received head_redis_address= {head_redis_address}')

    logging.info(f"Worker on rank {rank} with ip {fetch_ip()} will connect to head-redis-address={head_redis_address}")
    run_ray_worker(head_redis_address)

    comm.barrier() # waiting for all workers to start

if __name__ == "__main__":

    logging.basicConfig(
        filename='start_ray.log',
        format='%(asctime)s | %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)

    # print(rank)

    if rank == 0: 
        head_redis_address = master()
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
                        stdout=fp,
                        stderr=subprocess.STDOUT,
                        check=True,
        )
        logging.info("RL LIB invoked successfully. Exiting.")

    comm.barrier()
    os.system('ray stop')
    
    print('Successful exit')


