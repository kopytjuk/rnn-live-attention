#!/bin/bash
# training parameters
seq_len=100
epochs=1000
patience=100
# set experiment directory, will be created within the python code
current_ts=$(date +'%Y%m%d-%H%M%S')
experiment_dir="./experiments/$current_ts"

#run training
python.exe ./main.py $experiment_dir -e $epochs -p $patience -l $seq_len &
tensorboard.exe --logdir=$experiment_dir
