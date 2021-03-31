#!/bin/bash
#SBATCH --job-name=ngrok
#SBATCH --error=scripts/output/ngrok_err.err
#SBATCH --output=scripts/output/ngrok_out.log
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
sh ~/my/load_modules.sh
source deactivate
source activate voice


srun sh ./scripts/get_url.sh
