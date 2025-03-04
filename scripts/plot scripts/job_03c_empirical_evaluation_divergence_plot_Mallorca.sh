#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=40:00:00
#SBATCH --job-name=ES_1

source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env_new
python 03c_empirical_evaluation_divergence_plot.py --mallorca --days 90