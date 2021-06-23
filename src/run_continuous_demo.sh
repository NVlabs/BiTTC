#!/usr/bin/env bash

# Generate continuous TTC and OF estimation.
# Note: Our approach is not refined for accurate optical flow estimation. 
# We use optical flow estimation as an auxiliary task while training.
CUDA_VISIBLE_DEVICES=0 python run_continuous_estimation.py \
    --attributes ttc of \
    --pretrained '../weights/cont_ttc_kitti15_trainval.pth.tar' \
    --alpha_min 0.5 \
    --alpha_max 1.3 \
    --alpha_size 72 \
    --shift_min -48 \
    --shift_max 48 \
    --shift_delta 1 \
    --bittcnet_crop_height 384 \
    --bittcnet_crop_width 1152 \
    --ref_img_path '../data/img_ref.png' \
    --src_img_path '../data/img_src.png'
