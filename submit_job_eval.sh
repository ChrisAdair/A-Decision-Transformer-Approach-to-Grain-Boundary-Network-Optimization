#!/bin/bash --login

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --mail-user=cwa367@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
conda activate TorchEnv
module load spack
module load matlab
export LD_LIBRARY_PATH=${MATLABROOT}/runtime/glnxa64:${MATLABROOT}/bin/glnxa64:${MATLABROOT}/sys/os/glnxa64:${MATLABROOT}/sys/opengl/lib/glnxa64
export XAPPLRESDIR=${MATLABROOT}/X11/app-defaults
python evaluate.py