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
#SBATCH --gres=gpu:2
#SBATCH --output=logs/EXP2-LoRA-int8-.%j.out
#SBATCH --error=logs/EXP2-LoRA-int8-.%j.err

module load conda

source activate irp

python3 /mainfs/lyceum/lhb1g20/codellama-finetune/finetuner.py
