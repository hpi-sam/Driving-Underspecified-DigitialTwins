#!/usr/bin/env bash


ckpt=$1
num_chunk=$2
cur_ind=$3
src_dir=$4
out_dir=$5

common_command="
python encode.py \
    --data_path ${src_dir} \
    --ckpt ${ckpt} \
    --size 256 \
    --dataset carla \
    --results_path ${out_dir} \
    --test 1 \
    --num_div_batch 4
"


CUDA_VISIBLE_DEVICES=0 ${common_command} --num_chunk ${num_chunk} --cur_ind $((cur_ind)) &

## use multiple commands to parallelize e.g.
# CUDA_VISIBLE_DEVICES=0 ${common_command} --num_chunk ${num_chunk} --cur_ind $((cur_ind)) &
# CUDA_VISIBLE_DEVICES=1 ${common_command} --num_chunk ${num_chunk} --cur_ind $((cur_ind+1)) &
# CUDA_VISIBLE_DEVICES=2 ${common_command} --num_chunk ${num_chunk} --cur_ind $((cur_ind+2)) &
# CUDA_VISIBLE_DEVICES=3 ${common_command} --num_chunk ${num_chunk} --cur_ind $((cur_ind+3)) &
