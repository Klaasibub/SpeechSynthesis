#!/bin/bash

#SBATCH --job-name=apex_install				# Название задачи 
#SBATCH --error=SBATCH_apex_err.err			# Файл для вывода ошибок 
#SBATCH --output=SBATCH_apex_output.log			# Файл для вывода результатов 
#SBATCH --time=0:30:00					# Максимальное время выполнения
#SBATCH --cpus-per-task=8				# Количество CPU на одну задачу
#SBATCH --gpus=1					# Требуемое кол-во GPU
sh ~/load_modules.sh
source deactivate
source activate voice
echo «Assigned GPUs: $CUDA_VISIBLE_DEVICES»

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

