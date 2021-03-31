#!/bin/bash
#SBATCH --job-name=flask
#SBATCH --error=scripts/output/flask_err.err
#SBATCH --output=scripts/output/flask_out.log
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
sh ~/my/load_modules.sh
source deactivate
source activate voice

srun python flask_ngrok.py
