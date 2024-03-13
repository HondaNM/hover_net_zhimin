#!/bin/bash

input_dir=$1
output_dir=$2

python run_infer.py \
--gpu='4' \
--nr_types=0 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=original \
--model_path=/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/pretrained/hovernet_original_consep_notype_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=8 \
tile \
--input_dir="${input_dir}" \
--output_dir="${output_dir}" \
--mem_usage=0.2 \
--draw_dot
#--save_qupath
