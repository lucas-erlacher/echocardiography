#!/bin/bash

# Run this script:    bash /cluster/home/elucas/thesis/cluster/tune.sh

runs=5

use_gpu=true

python_script="/cluster/home/elucas/thesis/scripts/fine_tune.py"

# Job parameters
runtime="12:00:00"  # Having this too high can make the queue reduce your priority
cpus_per_task=4
gpu_type="rtx4090"
gpu_count=1
mem_per_cpu="32G"

good_seeds=(1 2 3 6 17)

# Ensure log dir exists
base_log_dir="/cluster/home/elucas/thesis/logs/"
mkdir -p "$base_log_dir"

# Create log_dir for this run
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
log_dir="${base_log_dir}/${timestamp}"
mkdir -p "$log_dir"

for (( run_idx=0; run_idx<$runs; run_idx++ )); do  
    seed=${good_seeds[$run_idx]}

    # GPU
    if [ "$use_gpu" = true ]; then
        sbatch -c $cpus_per_task -t $runtime --gres=gpu:$gpu_type:$gpu_count --mem-per-cpu=$mem_per_cpu -p "gpu" \
        --output="${log_dir}/console_output_run_${seed}.log" \
        --error="${log_dir}/errors_run_${seed}.log" \
        --job-name="tune_${seed}" \
        --ntasks-per-node=1 \
        --wrap="\
        bash -c 'source ~/.bashrc && \
        conda activate gpu && \
        python $python_script --run_idx $seed'" 
    # CPU
    else 
        sbatch -c $cpus_per_task -t $runtime --mem-per-cpu=$mem_per_cpu -p "compute" \
        --output="${log_dir}/console_output_run_${seed}.log" \
        --error="${log_dir}/errors_run_${seed}.log" \
        --job-name="tune_${seed}" \
        --ntasks-per-node=1 \
        --wrap="\
        bash -c 'source ~/.bashrc && \
        conda activate gpu && \
        python $python_script --run_idx $seed'" 
    fi

    sleep 1
done
