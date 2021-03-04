#!/bin/bash

#SBATCH --job-name=fastpitch_task			# Название задачи 
#SBATCH --error=SBATCH_prepare_err.err			# Файл для вывода ошибок 
#SBATCH --output=SBATCH_prepare_output.log		# Файл для вывода результатов 
#SBATCH --time=24:00:00					# Максимальное время выполнения
#SBATCH --cpus-per-task=16				# Количество CPU на одну задачу
#SBATCH --gpus=2					# Требуемое кол-во GPU
sh ~/load_modules.sh
source deactivate
source activate voice
echo «Assigned GPUs: $CUDA_VISIBLE_DEVICES»

srun sh ./scripts/prepare_libritts_dataset.sh		# Выполнение расчёта
