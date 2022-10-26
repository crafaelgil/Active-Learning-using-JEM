#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -N j100Blood
#$ -cwd 

source /etc/profile.d/modules.sh
module load singularitypro python/3.6/3.6.12 cuda/10.0/10.0.130.1 cudnn/7.4/7.4.2
singularity exec --bind $PWD:$HOME/workdir --pwd $HOME/workdir --nv docker://pytorch/pytorch:latest \
bash run_repeat_train_jem.sh
