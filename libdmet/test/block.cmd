#!/bin/bash

#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --time=0:9:50
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=128
#SBATCH --no-requeue
#SBATCH --constraint cpu
#SBATCH --mem=470G

source ~/.bashrc_perlmutter
ulimit -Sn 65536
ulimit -s unlimited
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

#python CCO.py
srun -n 4 --cpu-bind=cores -c 128 python test_spinless_model.py
