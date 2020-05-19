#!/bin/bash
#PBS -N test_script
#PBS -o test_out.log
#PBS -e test_err.log
#PBS -l select=1:ncpus=1
#PBS -q small
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
echo "user = "`echo $USER`
echo "project = "`echo $PROJECT`
echo "Finished at "`date`
