#!/bin/bash
#SBATCH --job-name=train
#SBATCH --error=scripts/output/train_err.err
#SBATCH --output=scripts/output/train_out.log
#SBATCH --time=9-24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gpus=1
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun python ./train.py -p=./data/hparams_home.yaml
