from mpi4py import MPI
import sys
import psutil

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write(
    "Hello, World! I am process %d of %d on %s.\n"
    % (rank, size, name))

sys.stdout.write('My CPU count is: %d \n' % (psutil.cpu_count(logical=False)))


