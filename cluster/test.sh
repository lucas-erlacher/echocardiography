#!/bin/bash

# run this script:                 bash /cluster/home/elucas/thesis/cluster/test.sh

num_folds=5

python_script="/cluster/home/elucas/thesis/scripts/test_folds.py"

# job parameters
runtime="12:00:00"  # having this too high can make the queue reduce your priority
cpus_per_task=4
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

for (( run_idx=0; run_idx<$num_folds; run_idx++ )); do  
    sbatch -c $cpus_per_task -t $runtime --mem-per-cpu=$mem_per_cpu -p "compute" \
    --output="/cluster/home/elucas/thesis/logs/${timestamp}/console_output_run_${run_idx}.log" \
    --error="/cluster/home/elucas/thesis/logs/${timestamp}/errors_run_${run_idx}.log" \
    --job-name="test_${run_idx}" \
    --ntasks-per-node=1 \
    --wrap="\
    bash -c 'source ~/.bashrc && \
    conda activate gpu && \
    python $python_script --run_idx $run_idx'"

    sleep 1
done