#!/bin/bash

####################################
#  Iridis 5 slurm script template  
#                                  
#  Submit script: sbatch filename  
#                                  
####################################
#SBATCH --ntasks=1     # Number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --cpus-per-task=2     # Threads per task
#SBATCH --time=60:00:00   # walltime
#SBATCH --partition=gpu # Name of the parititon
#SBATCH --gres=gpu:2
#SBATCH --output=logs/EVAL_QLoRA_step_300_13b.%j.out
#SBATCH --error=logs/EVAL_QLoRA_step_300_13b.%j.err

module load conda

source activate irp

cd /lyceum/lhb1g20/codellama-finetune/bigcode-evaluation-harness

accelerate launch main.py --model /mainfs/scratch/lhb1g20/CodeLlama-13b-hf/ --peft_model /mainfs/scratch/lhb1g20/checkpoints/EXPERIMENT-1-QLoRA-nf4-13B-bat22024-03-29-18h32m/checkpoint-300/ --load_in_4bit  --max_length_generation 512   --tasks humaneval  --temperature 0.2   --n_samples 10   --batch_size 2  --allow_code_execution
