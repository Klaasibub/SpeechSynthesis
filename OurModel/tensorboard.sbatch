#!/bin/bash
#SBATCH --job-name=board
#SBATCH --error=SBATCH_tensorboard_err.err
#SBATCH --output=SBATCH_tensorboard_out.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun tensorboard dev upload --logdir '/home/sidenko/my/logdir'
