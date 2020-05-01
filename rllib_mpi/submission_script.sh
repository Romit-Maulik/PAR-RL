#!/bin/bash
#COBALT -n 1
#COBALT -t 00:30:00
#COBALT -q debug-cache-quad 
#COBALT -A datascience

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
export PATH=/soft/datascience/anaconda3/bin:$PATH
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH
source activate rllib_env

aprun -n 4 -N 4 python start_ray.py

