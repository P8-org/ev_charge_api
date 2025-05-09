#!/bin/sh

#SBATCH --job-name=test  # Name of your job
#SBATCH --output=test-out.out     # Name of the output file
#SBATCH --error=file-test.err # Name of the error file
#SBATCH --mem=86G               # Memory
#SBATCH --cpus-per-task=64      # CPUs per task
#SBATCH --gres=gpu:5            # Allocated GPUs
#SBATCH --time=12:00:00         # Maximum run time


singularity exec --bind ~/my-virtual-env:/my-virtual-env /ceph/container/python/python_3.10.sif /bin/bash -c "source /my-virtual-env/bin/activate && python3 '/ceph/project/charge_buddy/actions-runner/_work/ev_charge_api/ev_charge_api/run_DQN.py'"
