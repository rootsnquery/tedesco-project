#!/bin/sh
#
# Baseline model for albedo project submit script for Slurm.
#
#SBATCH --account=dsi # The account name for the job.
#SBATCH --job-name=albedo_baseline_models_5per # The job name.
#SBATCH -N 1
#SBATCH --mem-per-cpu=20gb

module load anaconda
python3 -m pip install -U pip
pip install xgboost

#Command to execute Python program
python model.py > record_all_xgboost.txt

#End of script
