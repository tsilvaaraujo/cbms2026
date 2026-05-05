#!/bin/bash

job_file="clinical_job_list.txt"
rm -f "$job_file"

mkdir -p logs

for config_path in experiments/features_*; do
    config_name=$(basename "$config_path")
    num_features=$(echo "$config_name" | awk -F'_' '{print NF - 1}')

    for exec_id in {0..9}; do
        echo "$config_name $exec_id $num_features" >> "$job_file"
    done
done

total_jobs=$(wc -l < "$job_file")
echo "📄 Prepared $total_jobs jobs in $job_file"

sbatch --array=0-$((total_jobs - 1))%2 run_clinical_array.sh
