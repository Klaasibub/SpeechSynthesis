#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --error=SBATCH_err.err
#SBATCH --output=SBATCH_out.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun sh ./inference.sh -p=data/my_params.yaml
