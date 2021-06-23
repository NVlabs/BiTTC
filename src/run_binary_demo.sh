#!/usr/bin/env bash

# Generate binary TTC segmenetation and combine them to generate quantized TTC
CUDA_VISIBLE_DEVICES=0 python run_binary_estimation.py \
    --attribute ttc \
    --pretrained '../weights/binary_ttc_kitti15_train.pth.tar' \
    --fps 10 \
    --alpha_vals 0.7 0.75 0.8 0.85 0.9 0.95 0.98 1.02 1.10 \
    --ref_img_path '../data/img_ref.png' \
    --src_img_path '../data/img_src.png' 

# Generate binary optical flow segmentation along x and y directions
# Note: Our approach is not refined for accurate optical flow estimation. 
# We use optical flow estimation as an auxiliary task while training.
CUDA_VISIBLE_DEVICES=0 python run_binary_estimation.py \
    --attribute of \
    --pretrained '../weights/binary_ttc_kitti15_train.pth.tar' \
    --shifts_x -84 -64 -48 -36 -24 -12 0 12 24 36 48 64 84 \
    --shifts_y -36 -24 -12 0 6 12 18 24 30 48 60 72 84 \
    --ref_img_path '../data/img_ref.png' \
    --src_img_path '../data/img_src.png' 
