#!/bin/bash
#COBALT -t 180
#COBALT -n 128
#COBALT -q default
#COBALT -A datascience
#COBALT -M spawar@anl.gov
#COBALT --attrs enable_ssh=1:ssds=required:ssd_size=128

echo "Running Cobalt Job $COBALT_JOBID."

# Load OpenFOAM into your environment
source /home/projects/OpenFOAM/OpenFOAM-5.x/etc/bashrc.ThetaIcc

#..NOTE: you will need the following to run OF executables in parallel
#..^^^^  (do not comment the following line for parallel executables)
export LD_LIBRARY_PATH=${FOAM_LIBBIN}:${FOAM_LIBBIN}/mpi:${LD_LIBRARY_PATH}

export KMP_BLOCKTIME=0
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export OMP_PROC_BIND='spread,close'
export OMP_NESTED='TRUE'

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
module load intelpython36
#export PATH=/soft/datascience/anaconda3/bin:$PATH
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH
source activate rllib_env1_fenics

aprun -n $COBALT_JOBSIZE -N 1 --cc none python start_ray.py
