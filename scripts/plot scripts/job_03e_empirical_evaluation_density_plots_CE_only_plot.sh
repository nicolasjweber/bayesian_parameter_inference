#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=140G
#SBATCH --time=1:00:00
#SBATCH --job-name=CE_1

source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env_new
python 03e_empirical_evaluation_density_plots.py --continental_europe --days 90 --only_plot