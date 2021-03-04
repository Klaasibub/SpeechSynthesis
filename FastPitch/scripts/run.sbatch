#!/bin/bash

#SBATCH --job-name=fastpitch_task			# Название задачи 
#SBATCH --error=SBATCH_err.err			# Файл для вывода ошибок 
#SBATCH --output=SBATCH_output.log		# Файл для вывода результатов 
#SBATCH --time=48:00:00					# Максимальное время выполнения
#SBATCH --cpus-per-task=24				# Количество CPU на одну задачу
#SBATCH --gpus=2					# Требуемое кол-во GPU
sh ~/load_modules.sh
source deactivate
source activate voice
echo «Assigned GPUs: $CUDA_VISIBLE_DEVICES»

srun sh ./scripts/train.sh	# Выполнение расчёта
