#!/bin/bash

# run this script:                 bash /cluster/home/elucas/thesis/cluster/train.sh

cross_val=1
num_folds=10

use_gpu=true

python_script="/cluster/home/elucas/thesis/trainer.py"

# job parameters
runtime="12:00:00"  # having this too high can make the queue reduce your priority
cpus_per_task=4
gpu_type="rtx4090"
gpu_count=1
mem_per_cpu="32G"

# ensure log dir exists
base_log_dir="/cluster/home/elucas/thesis/logs/"
if [ ! -d "$base_log_dir" ]; then
    mkdir -p "$base_log_dir"
fi

# create log_dir for this run
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
log_dir="${base_log_dir}/${timestamp}"
mkdir -p "$log_dir"

for (( fold_idx=0; fold_idx<$num_folds; fold_idx++ )); do
    # GPU
    if [ "$use_gpu" = true ]; then
        sbatch -c $cpus_per_task -t $runtime --gres=gpu:$gpu_type:$gpu_count --mem-per-cpu=$mem_per_cpu -p "gpu" \
        --output="/cluster/home/elucas/thesis/logs/${timestamp}/console_output_fold_${fold_idx}.log" \
        --error="/cluster/home/elucas/thesis/logs/${timestamp}/errors_fold_${fold_idx}.log" \
        --job-name="fold_${fold_idx}" \
        --ntasks-per-node=1 \
        --wrap="\
        bash -c 'source ~/.bashrc && \
        conda activate gpu && \
        python $python_script --cross_val $cross_val --fold_idx $fold_idx --num_folds $num_folds'"
    # CPU
    else  
        sbatch -c $cpus_per_task -t $runtime --mem-per-cpu=$mem_per_cpu -p "compute" \
        --output="/cluster/home/elucas/thesis/logs/${timestamp}/console_output_fold_${fold_idx}.log" \
        --error="/cluster/home/elucas/thesis/logs/${timestamp}/errors_fold_${fold_idx}.log" \
        --job-name="fold_${fold_idx}" \
        --ntasks-per-node=1 \
        --wrap="\
        bash -c 'source ~/.bashrc && \
        conda activate gpu && \
        python $python_script --cross_val $cross_val --fold_idx $fold_idx --num_folds $num_folds'"
    fi

    sleep 1
done