#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=train

source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env_new
python 01_train.py --parameter_estimation_daytime_specific "KRUSE plot data/prior_parameter_ranges.csv" --time_span 3600 --dt 1