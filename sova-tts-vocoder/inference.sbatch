#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --error=SBATCH_inference_err.err
#SBATCH --output=SBATCH_inference_out.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun python3 inference.py -f=original.txt -w=/storage/sidenko/voco_checkpoints/waveglow_21000 -o=examples
