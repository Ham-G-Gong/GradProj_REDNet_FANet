#!/bin/bash -l

#$ -N FA1_Res18_CCRI

#$ -m bea

#$ -M pinghu@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 5

# Request 2 GPU
#$ -l gpus=1
#$ -l gpu_type=A6000|RTX8000

# Specify the minimum GPU compute capability
#$ -l gpu_c=6.1

#$ -l h_rt=240:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="



# module load miniconda
# module load cuda/11.3
# module load gcc
# #conda activate py37
# #conda install -c conda-forge opencv

# export PROJECT_PATH=/projectnb/ivc-ml

# cd $PROJECT_PATH/pinghu/code/lpcvc/

# nvidia-smi

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=44248  train.py --config ./configs/FA_Res18.yml
