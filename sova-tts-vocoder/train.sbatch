#!/bin/bash
#SBATCH --job-name=vocoder_task
#SBATCH --error=SBATCH_err.err
#SBATCH --output=SBATCH_out.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=2
sh ~/my/load_modules.sh
source deactivate
source activate voice

echo Assigned GPUs: $CUDA_VISIBLE_DEVICES

srun python3 distributed.py -c=config.json
