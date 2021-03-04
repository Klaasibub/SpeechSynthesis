#!/bin/bash

#SBATCH --job-name=inference				# Название задачи 
#SBATCH --error=SBATCH_inference_err.err		# Файл для вывода ошибок 
#SBATCH --output=SBATCH_inference_output.log		# Файл для вывода результатов 
#SBATCH --time=4:00:00					# Максимальное время выполнения
#SBATCH --cpus-per-task=16				# Количество CPU на одну задачу
#SBATCH --gpus=2					# Требуемое кол-во GPU
sh ~/load_modules.sh
source deactivate
source activate voice
echo «Assigned GPUs: $CUDA_VISIBLE_DEVICES»

srun sh ./scripts/inference_example.sh	# Выполнение расчёта
