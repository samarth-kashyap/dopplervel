#!/bin/bash
#PBS -N lct.2010
#PBS -o lctout1.log
#PBS -e lcterr1.log
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=01:30:00
#PBS -q small
echo "Starting at "`date`
cd /home/g.samarth/dopplervel2/
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
echo $PBS_JOBID
parallel --jobs 32 < /home/g.samarth/dopplervel2/jobscripts/ipjobs_lct_2010.sh
echo "Finished at "`date`
