#!/bin/bash
#COBALT -t 60
#COBALT -n 1
#COBALT -q debug-cache-quad
#COBALT -A datascience
#COBALT -M spawar@anl.gov
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
source activate rllib_env1_fenics

aprun -n $COBALT_JOBSIZE -N 1 --cc none python start_ray.py
