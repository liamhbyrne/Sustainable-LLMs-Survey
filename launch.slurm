#!/bin/bash

####################################
#  Iridis 5 slurm script template  
#                                  
#  Submit script: sbatch filename  
#                                  
####################################
#SBATCH --ntasks=1     # Number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --cpus-per-task=4     # Threads per task
#SBATCH --time=60:00:00   # walltime
#SBATCH --partition=lyceum # Name of the parititon
#SBATCH --gres=gpu:1
#SBATCH --output=logs/LOG.%j.out
#SBATCH --error=logs/LOG.%j.err

module load conda
source activate irp
python3 finetuner.py >> /scratch/lhb1g20/run.log