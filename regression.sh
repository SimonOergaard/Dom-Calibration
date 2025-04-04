#!/bin/bash

#SBATCH --job-name=extraction    # Job name
#SBATCH --output=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.out              # Standard output
#SBATCH --error=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.err               # Error log
#SBATCH --partition=gr10_gpu               # GPU partition
#SBATCH --cpus-per-task=20                 # Number of CPU cores per task
#SBATCH --mem=32gb                         # Memory limit
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --nodes=1                        # Request one node
#SBATCH --mail-type=END,FAIL              # Notifications for job done & fail
#SBATCH --mail-user=jxt726@alumni.ku.dk  # Email for notifications


source ~/Icecube/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cp /groups/icecube/simon/GNN/workspace/Scripts/filtered_all_big_data.db ${SCRATCH}/
# Run the Python script with GPU usage
#srun python regression.py train
srun python regression.py predict --weights best_weights
