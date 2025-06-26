#!/bin/bash
#SBATCH --job-name=min_max # Job name
#SBATCH --output=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.out              # Standard output
#SBATCH --error=/groups/icecube/simon/GNN/workspace/ppc_log/ppc_%j.err         
#SBATCH --partition=icecube              # Use GPU partition

#SBATCH --ntasks=1                       # Single task
#SBATCH --mail-type=END,FAIL             # Mail events
#SBATCH --mail-user=jxt726@alumni.ku.dk  # Email for notifications
#SBATCH --cpus-per-task=40               # Number of CPU cores per task
#SBATCH --nodes=1                        # Request one node
#SBATCH --time=08:30:00                  # Time limit
#SBATCH --mem=240G                        # Memory request

# Activate your environment
#export PATH=$HOME/ffmpeg/bin:$PATH
source ~/Icecube/bin/activate   # Adjust to the correct path of your environment
# Run the Python script
#python Burn_sample_check.py
#python extractor_test.py sqlite icecube-upgrade
#python data_sorter.py
#python String_no_bin.py
#python line_fit.py
python Ride_new_new.py
#python Ride_analysis.py
#python muon_sorter.py
#python RIDE_new.py
#python RIDE_time.py
#python RIDE_min_max.py
#python db_check.py
#python String_look_up.py