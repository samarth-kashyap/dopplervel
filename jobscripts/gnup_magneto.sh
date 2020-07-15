#!/bin/bash
#PBS -N inv_magneto
#PBS -o inv_mag_out.log
#PBS -e inv_mag_err.log
#PBS -l place=vscatter -l select=10:ncpus=1
#PBS -q large
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_WORKDIR
echo $PBS_JOBID
source activate /home/g.samarth/anaconda3/envs/py3
parallel --jobs 1 --sshloginfile $PBS_NODEFILE --workdir /home/g.samarth/dopplervel2/jobscripts < /home/g.samarth/dopplervel2/jobscripts/ipjobs_magneto.sh
echo "Finished at "`date`
