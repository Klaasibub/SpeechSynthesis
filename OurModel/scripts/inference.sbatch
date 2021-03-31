#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --error=scripts/output/inference_err.err
#SBATCH --output=scripts/output/inference_out.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun python ./inference.py -p=data/my_params.yaml -c /home/sidenko/my/output/23k
