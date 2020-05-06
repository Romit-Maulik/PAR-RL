# Minimum working example to run Ray error on Theta:

## Introduction to Ray and distributed RL

Based on the template provided [here](https://github.com/ytopt-team/tuster/blob/master/tuster/system/theta/run.py)
I was able to set up a reinforcement learning search using RLLib and Ray. The idea is as follows:

1. Start ray on the head node and retrieve the `head_redis_address` by running
```
ray start --num_cpus 1 --redis-port=10100
```
note that the `--num_cpus` argument may vary from machine to machine.

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

4. All this business can be packaged quite effectively using `start_ray.py` which uses subprocess to call `train_ppo.py`.

5. This distributed RL runs without any trouble at all on my laptop for 4 workers and can be called by running `mpirun -np 4 python start_ray.py`

## Running on Theta compute nodes

When I try to execute the same procedure by running an interactive job on Theta using
```
qsub -A datascience -t 60 -q debug-cache-quad -n 2 -I
```
and using aprun as `aprun -n 4 -N 2 python start_ray.py`. The logs for starting ray (at `start_ray.log`) show success:

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
