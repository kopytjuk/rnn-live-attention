#!/bin/bash
# training parameters
seq_len=100
epochs=100
patience=10
# set experiment directory, will be created within the python code
current_ts=$(date +'%Y%m%d-%H%M%S')
experiment_dir="./experiments/$current_ts"

#run training
python.exe ./main.py $experiment_dir -e $epochs -p $patience -l $seq_len
