#!/bin/bash
#SBATCH --job-name=board
#SBATCH --error=scripts/output/tensorboard_err.err
#SBATCH --output=scripts/output/tensorboard_out.log
#SBATCH --time=10-24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun tensorboard dev upload --logdir '/home/zolkin/logdir'
