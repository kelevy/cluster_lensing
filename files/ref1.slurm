#!/bin/bash
# Partition for the job:
#SBATCH --partition=physical
# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1
# The name of the job:
#SBATCH --job-name="ref1"
# The project ID which this job should run under:
#SBATCH --account="punim1922"
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# The amount of memory in megabytes per node:
#SBATCH --mem=100000
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Loading modules

# Launch python code
python ref1.py 


# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
