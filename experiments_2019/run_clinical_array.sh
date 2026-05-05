#!/bin/bash
#SBATCH --job-name=clinical_data
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=42:00
#SBATCH --partition=blaise
#SBATCH --nodelist=blaise

JOB_LIST="clinical_job_list.txt"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

config_name=$(echo "$line" | awk '{print $1}')
exec_id=$(echo "$line" | awk '{print $2}')
num_features=$(echo "$line" | awk '{print $3}')

echo "🚀 Dispatching config=$config_name | exec_id=$exec_id | num_features=$num_features"

srun --export=ALL,CONFIG_NAME="$config_name",EXEC_ID="$exec_id",NUM_FEATURES="$num_features" bash run_clinical.sh
