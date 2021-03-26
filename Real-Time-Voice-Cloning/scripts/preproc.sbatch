#!/bin/bash
#SBATCH --job-name=prerpocess
#SBATCH --error=scripts/output/preprocess_err.err
#SBATCH --output=scripts/output/preprocess_out.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gpus=0
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun python synthesizer_preprocess_audio.py ~/my --no_alignments 
