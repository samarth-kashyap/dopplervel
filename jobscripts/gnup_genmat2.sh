#!/bin/bash
#PBS -N gen_mat2
#PBS -o genmatout2.log
#PBS -e genmaterr2.log
#PBS -l place=vscatter -l select=10:ncpus=1
#PBS -q large
echo "Starting at "`date`
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_WORKDIR
echo $PBS_JOBID
source activate /home/g.samarth/anaconda3/envs/py3
parallel --jobs 1 --sshloginfile $PBS_NODEFILE --workdir /home/g.samarth/dopplervel2 < /home/g.samarth/dopplervel2/ipjobs_genmat2.sh
echo "Finished at "`date`
