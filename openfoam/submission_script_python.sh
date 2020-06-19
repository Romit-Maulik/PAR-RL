#!/bin/bash -f
#COBALT -t 00:30:00
#COBALT -n 1
#COBALT -A datascience
#COBALT -q debug-cache-quad
#COBALT -M spawar@anl.gov

# Load OpenFOAM into your environment
source /home/projects/OpenFOAM/OpenFOAM-5.x/etc/bashrc.ThetaIcc

#..NOTE: you will need the following to run OF executables in parallel
#..^^^^  (do not comment the following line for parallel executables)
export LD_LIBRARY_PATH=${FOAM_LIBBIN}:${FOAM_LIBBIN}/mpi:${LD_LIBRARY_PATH}

export KMP_BLOCKTIME=0
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export OMP_PROC_BIND='spread,close'
export OMP_NESTED='TRUE'

solver="simpleFoam"
#solverOptions="-parallel"
solveroptions=""
now=$(date "+%m.%d.%Y-%H.%M.%S")
solverLogFile="log.${solver}-${now}"

rpn=1 #32
hpc=1
nodes=$COBALT_JOBSIZE
OMP_Threads=128

aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} python parameter_change_v2.py

#aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} -cc depth -d ${OMP_Threads} --env OMP_NUM_THREADS=${OMP_Threads} valgrind --tool=callgrind $FOAM_APPBIN/${solver}  >> ${solverLogFile}
#aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} -cc depth -d ${OMP_Threads} --env OMP_NUM_THREADS=${OMP_Threads} $FOAM_USER_APPBIN/${solver}  >> ${solverLogFile}
#aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} -cc depth -d 1 --env OMP_NUM_THREADS=${OMP_Threads} \
#amplxe-cl -collect advanced-hotspots -analyze-system -finalization-mode=none -source-search-dir=/gpfs/mira-home/hsharma/OpenFOAM/hsharma-5.x/platforms \
#-r vtune-result-dir_hotspots_${OMP_Threads}/ $FOAM_USER_APPBIN/${solver}  >> ${solverLogFile}

#Base=$(pwd)
#cd /projects/datascience/hsharma/Practice/affinity
#aprun -n $((nodes*rpn)) -N ${rpn} -j ${hpc} -cc depth -d ${OMP_Threads} --env OMP_NUM_THREADS=${OMP_Threads} ./hello_affinity
#cd ${Base}


#echo $OMP_NUM_THREADS

