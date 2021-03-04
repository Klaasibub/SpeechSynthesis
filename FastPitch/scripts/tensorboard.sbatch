#!/bin/bash

#SBATCH --job-name=tensorboard				# Название задачи 
#SBATCH --error=SBATCH_tensorboard_err.err		# Файл для вывода ошибок 
#SBATCH --output=SBATCH_tensorboard_output.log		# Файл для вывода результатов 
#SBATCH --time=48:00:00					# Максимальное время выполнения
sh ~/load_modules.sh
source deactivate
source activate voice

srun tensorboard dev upload --logdir '/home/zolkin/output/train'
