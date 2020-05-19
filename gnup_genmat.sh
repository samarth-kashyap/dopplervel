#!/bin/bash
#PBS -N gen_mat1
#PBS -o genmatout.log
#PBS -e genmaterr.log
#PBS -l place=vscatter -l select=10:ncpus=1
#PBS -q large
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_WORKDIR
echo $PBS_JOBID
source activate /home/g.samarth/anaconda3/envs/py3
parallel --jobs 1 --sshloginfile $PBS_NODEFILE --workdir /home/g.samarth/dopplervel2 < /home/g.samarth/dopplervel2/ipjobs_genmat3.sh
echo "Finished at "`date`
