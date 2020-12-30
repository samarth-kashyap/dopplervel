#!/bin/bash
#PBS -N HMI_ME720s
#PBS -o vecmagout.log
#PBS -e vecmagerr.log
#PBS -l place=vscatter -l select=10:ncpus=1
#PBS -q large
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_WORKDIR
echo $PBS_JOBID
source activate /home/g.samarth/anaconda3/envs/py3
parallel --jobs 1 --sshloginfile $PBS_NODEFILE --workdir /home/g.samarth/dopplervel2/jobscripts < /home/g.samarth/dopplervel2/jobscripts/ipjobs_vecmagneto.sh
echo "Finished at "`date`
