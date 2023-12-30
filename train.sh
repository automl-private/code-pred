#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres=gpu:1
# #SBATCH --mem 65000 # memory pool for each core (4GB)
#SBATCH -t 5-23:01 # time (D-HH:MM)
# #SBATCH -c 8 # number of cores
#SBATCH -o /home/muellesa/log/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /home/muellesa/log/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J code-pred-job # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Executing $COMMAND";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "On node ${hostname}";
. /home/muellesa/use_env.sh code-pred-310;
cd ~/merge_test/code-pred;

eval python -m train.py "$@";

#rm -r $tmpdir;

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";