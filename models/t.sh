#!/bin/sh

#SBATCH --job-name=test  # Name of your job
#SBATCH --output=test/test-out.out     # Name of the output file
#SBATCH --error=test/file-test.err # Name of the error file
#SBATCH --mem=24G               # Memory
#SBATCH --cpus-per-task=15      # CPUs per task
#SBATCH --gres=gpu:3            # Allocated GPUs
#SBATCH --time=01:00:00         # Maximum run time

singularity exec --bind ~/my-virtual-env:/my-virtual-env /ceph/container/python/python_3.10.sif /bin/bash -c "source /my-virtual-env/bin/activate && python test1.py"
#singularity exec --nv /ceph/container/python/python_3.10.sif python3 test1.py
#python test1.py
