#!/bin/sh
#
# Baseline model for albedo project submit script for Slurm.
#
#SBATCH --account=dsi # The account name for the job.
#SBATCH --job-name=albedo_baseline_models # The job name.
#SBATCH -N 1
#SBATCH --mem-per-cpu=100gb

module load anaconda


#Command to execute Python program
python model.py > albedo_df_updated/XGboost_record.txt

#End of script
