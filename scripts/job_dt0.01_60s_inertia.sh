#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=train

source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env_new
python 01_train.py --inertia_estimation