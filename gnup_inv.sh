#!/bin/bash
#PBS -N invHMI
#PBS -o inv_out.log
#PBS -e inv_err.log
#PBS -l select=1:ncpus=32
#PBS -q small
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
parallel --jobs 32 < /home/g.samarth/dopplervel2/ipjobs_inv.sh
echo "Finished at "`date`
