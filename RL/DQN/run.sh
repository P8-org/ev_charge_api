#!/bin/bash

#SBATCH --job-name=test  # Name of your job
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=test-out.out     # Name of the output file
#SBATCH --error=file-test.err # Name of the error file
#SBATCH --mem=64G               # Memory
#SBATCH --cpus-per-task=32      # CPUs per task
#SBATCH --gres=gpu:5            # Allocated GPUs
#SBATCH --time=11:59:00         # Maximum run time
#SBATCH --requeue

#######################################################################################
# tweak this to fit your needs
max_restarts=20

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Dynamically set output and error filenames using job ID and iteration
outfile="${SLURM_JOB_ID}_${iteration}.out"
errfile="${SLURM_JOB_ID}_${iteration}.err"

# Print the filenames for debugging
echo "Output file: ${outfile}"
echo "Error file: ${errfile}"

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
        # Requeue the job, allowing it to restart with incremented iteration
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM

#######################################################################################

srun --output="${outfile}" --error="${errfile}" singularity exec --bind ~/my-virtual-env:/my-virtual-env /ceph/container/python/python_3.10.sif /bin/bash -c "source /my-virtual-env/bin/activate && python3 '/ceph/project/charge_buddy/actions-runner/_work/ev_charge_api/ev_charge_api/run_DQN.py'"
