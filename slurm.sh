#!/bin/bash
#SBATCH --job-name=RIDE_calculation       # Job name
#SBATCH --output=ppc_%j.out       # Standard output
#SBATCH --error=ppc_%j.err        # Error log
#SBATCH --partition=icecube   # Use GPU partition
#SBATCH --ntasks=1                # Single task
#SBATCH --mail-type=END,FAIL      # Mail events
#SBATCH --mail-user=jxt726@alumni.ku.dk
#SBATCH --cpus-per-task=20         # One CPU core
#SBATCH --nodes=2                 # Request one node
#SBATCH --time=08:30:00           # Time limit
#SBATCH --mem=150G                 # Memory request

activate_env
python RIDE_new.py
logfile = "/home/simon/GNN/workspace/Scripts/slurm-{}.out".format(os.environ["SLURM_JOB_ID"])
exec >dev/null 2>$logfile