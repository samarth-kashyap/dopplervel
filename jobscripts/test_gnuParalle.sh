#!/bin/bash
#PBS -N cleanHMI
#PBS -o test-parallel-out
#PBS -e test-parallel-error
#PBS -l select=1:ncpus=8
#PBS -q small
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
parallel --jobs 8 < /home/g.samarth/dopplervel2/inputjobs.sh
echo "Finished at "`date`
