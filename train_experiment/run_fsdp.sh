#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --output=train_experiment/logs/fsdp_train_%j.log
#SBATCH --error=train_experiment/logs/fsdp_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Initialize conda
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda activate fsdp

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0

# Run the training script with srun
cd train_experiment
srun python train_fsdp.py 