#!/bin/bash -x
ENVIRON_BASE=$(dirname $(dirname $(which python)))
module swap PrgEnv-intel PrgEnv-gnu
mkdir _mpi4py-build
cd _mpi4py-build
git clone https://github.com/cython/cython.git
cd cython
git checkout tags/0.25.2 # later tags seem to break (haven't investigated further)
CC=cc CXX=CC python setup.py build
CC=cc CXX=CC python setup.py install --prefix=$ENVIRON_BASE
cd ..
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py
# mpi.cfg replacement
sed -i "s/## mpicc .*/mpicc = cc/g" mpi.cfg
sed -i "s/## mpicxx .*/mpicxx = CC/g" mpi.cfg
# setup.cfg replacement
sed -i "s/# mpicc .*/mpicc = cc/g" setup.cfg
sed -i "s/# mpicxx .*/mpicxx = CC/g" setup.cfg
sed -i "s/# mpifort .*/mpifort = ftn/g" setup.cfg
sed -i "s/# mpif90 .*/mpif90 = ftn/g" setup.cfg
sed -i "s/# mpif77 .*/mpif77 = ftn/g" setup.cfg
CC=cc CXX=CC python setup.py build
CC=cc CXX=CC python setup.py install --prefix=$ENVIRON_BASE
cd ..
rm -r _mpi4py-build