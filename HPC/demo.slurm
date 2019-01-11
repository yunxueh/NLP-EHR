#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --time=00:5:00
#SBATCH --partition=physical

#Load required modules
module load Python/3.6.4-intel-2017.u2
module load Tensorflow/1.9.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
pip install --user keras
pip install --user sklearn


#Launch multiple process python code
time mpiexec python MPI.py 