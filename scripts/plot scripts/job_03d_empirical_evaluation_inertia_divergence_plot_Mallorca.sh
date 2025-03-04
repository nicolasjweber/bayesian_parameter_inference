#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --job-name=ES_2

source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate sbi_env_new
python 03d_empirical_evaluation_inertia_divergence_plot.py --mallorca --days 90