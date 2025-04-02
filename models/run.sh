#!/bin/sh

#SBATCH --job-name=test  # Name of your job
#SBATCH --output=test-out.out     # Name of the output file
#SBATCH --error=file-test.err # Name of the error file
#SBATCH --mem=24G               # Memory
#SBATCH --cpus-per-task=15      # CPUs per task
#SBATCH --gres=gpu:3            # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time

singularity exec --bind /ceph/project/charge_buddy/my-virtual-env:/my-virtual-env /ceph/container/python/python_3.10.sif /bin/bash -c "source /my-virtual-env/bin/activate && python3 ./run_RL.py"
