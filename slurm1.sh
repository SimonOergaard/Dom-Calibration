#!/bin/bash
#SBATCH --job-name=extractor  # Job name
#SBATCH --output=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.out              # Standard output
#SBATCH --error=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.err         
#SBATCH --partition=icecube              # Use GPU partition

#SBATCH --ntasks=1                       # Single task
#SBATCH --mail-type=END,FAIL             # Mail events
#SBATCH --mail-user=jxt726@alumni.ku.dk  # Email for notifications
#SBATCH --cpus-per-task=20               # Number of CPU cores per task
#SBATCH --nodes=1                        # Request one node
#SBATCH --time=14:30:00                  # Time limit
#SBATCH --mem=120G                        # Memory request

# Activate your environment
source ~/Icecube/bin/activate   # Adjust to the correct path of your environment

# Run the Python script
#python Ride_new_new.py
#python Burn_sample_check.py
#python linefit_i3_writer.py
#python String_no_bin.py

python extractor_test.py sqlite icecube-86