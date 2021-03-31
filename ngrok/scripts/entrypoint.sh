#!/bin/bash

sbatch scripts/run_flask.sbatch
sbatch scripts/start_ngrok.sbatch
srun python -u scripts/endless.py
