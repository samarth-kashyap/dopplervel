#!/bin/bash
#PBS -N rotSpec
#PBS -o rot2out.log
#PBS -e rot2err.log
#PBS -l place=vscatter -l select=2:ncpus=8
#PBS -q large
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_WORKDIR
echo $PBS_JOBID
parallel --jobs 8 --sshloginfile $PBS_NODEFILE --workdir /home/g.samarth/dopplervel2/jobscripts < /home/g.samarth/dopplervel2/jobscripts/ipjobs_rot.sh
echo "Finished at "`date`
