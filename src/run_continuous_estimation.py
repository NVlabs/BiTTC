# Copyright 2021 NVIDIA CORPORATION & AFFILIATES
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import torch
import torchvision.transforms as transforms
import models
import cv2
import sys
import numpy as np
import datetime
import random
import math

from util import disp2rgb, str2bool, flow_uv_to_colors
from geometry_utils import prepare_warping_transforms, compute_transformed_volume
from geometry_utils import compute_volume_generation_params


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(allow_abbrev=False)

# Model
parser.add_argument('--ref_img_path',            type=str,      default='../data/img_ref.png')
parser.add_argument('--src_img_path',            type=str,      default='../data/img_src.png')
parser.add_argument('--attributes',              type=str,      nargs='*', default=['ttc', 'of'])

parser.add_argument('--pretrained',              type=str,      default='../models/cont_ttc_kitti15_trainval.tar')
parser.add_argument('--arch',                    type=str,      default='bittcnet_continuous_of_ttc_2d')
parser.add_argument('--bittcnet_featnet_arch',   type=str,      default='featextractnetspp')
parser.add_argument('--bittcnet_segnet_arch',    type=str,      default='segnet2d')
parser.add_argument('--segnet_num_imgs',         type=int,      default=2)
parser.add_argument('--segnet_num_segs',         type=int,      default=3)
parser.add_argument('--segnet_is_deep',          type=str2bool, default=True)
parser.add_argument('--regrefinenet_out_planes', type=int,      default=32)

parser.add_argument('--bittcnet_max_scale',      type=float,    default=1.5) 
parser.add_argument('--alpha_min',               type=float,    default=0.5)
parser.add_argument('--alpha_max',               type=float,    default=1.3)
parser.add_argument('--alpha_size',              type=int,      default=72)
parser.add_argument('--shift_min',               type=float,    default=-72.0)
parser.add_argument('--shift_max',               type=float,    default=72.0)
parser.add_argument('--shift_delta',             type=int,      default=1)
parser.add_argument('--bittcnet_crop_height',    type=int,      default=384)
parser.add_argument('--bittcnet_crop_width',     type=int,      default=1152)


####################################################################################################
args, unknown = parser.parse_known_args()
options = vars(args)

device = torch.device("cuda")

options['dtype'] = torch.float32
options['device'] = device
####################################################################################################


def main():

    print("==> ALL PARAMETERS")
    for key in options:
        print("{} : {}".format(key, options[key]))

    out_dir = "results"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    timestamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

    out_dir = "./results/%s_cont_%s"%(timestamp, '-'.join(args.attributes))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    print('\nResults will be dumped in %s'%out_dir)

    
    ################################################################################################
    # DATA
    ref_img = cv2.imread(args.ref_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 255.0
    src_img = cv2.imread(args.src_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 255.0

    h, w, _c = ref_img.shape

    # Ensure all the tensor dimensions are integers
    # The following code assumes bittcnet_max_scale as 1.5
    H_full = int(math.ceil(h / 192)) * 192 
    W_full = int(math.ceil(w / 192)) * 192
    options['bittcnet_full_height'] = H_full
    options['bittcnet_full_width'] = W_full
    y1 = int(H_full / 2 - h / 2)
    x1 = int(W_full / 2 - w / 2)

    print('\nPadding will be added to ensure input image size divisible by 192')
    print('New input image size is %d x %d'%(H_full, W_full))

    # Read images
    ref_img = torch.Tensor(ref_img).type(torch.float32).permute(2, 0, 1)
    ref_img = transforms.functional.normalize(ref_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ref_img_big = torch.zeros(3, H_full, W_full).type(torch.FloatTensor).cuda()
    ref_img_big[:, int(H_full/2-h/2):int(H_full/2+h/2), int(W_full/2-w/2):int(W_full/2+w/2)] = ref_img
    
    src_img = torch.Tensor(src_img).type(torch.float32).permute(2, 0, 1)
    src_img = transforms.functional.normalize(src_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    src_img_big = torch.zeros(3, H_full, W_full).type(torch.FloatTensor).cuda()
    src_img_big[:, int(H_full/2-h/2):int(H_full/2+h/2), int(W_full/2-w/2):int(W_full/2+w/2)] = src_img
    
    # prepare the image dimensions
    H_out = args.bittcnet_crop_height
    W_out = args.bittcnet_crop_width
    print('Crop size will be %d x %d'%(H_out, W_out))
    H_in = int(args.bittcnet_max_scale * H_out)
    W_in = int(args.bittcnet_max_scale * W_out)
    assert (H_out % 192 == 0 and W_out % 192 == 0), 'crop dimensions should be divisible by 192'
    

    # prepare the plane sweep volume lists
    shift_delta = args.shift_delta
    assert (args.alpha_size % 3 == 0), 'TTC volume size should be multiple of 3'
    assert (args.shift_max % (3 * shift_delta) == 0 and args.shift_min % (3 * shift_delta) == 0), \
        'OF volume size should be multiple of 3'
    alpha_delta = (args.alpha_max - args.alpha_min) / (args.alpha_size)

    ####################################################################################################
    # MODEL
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](options, network_data).cuda().eval()
        
    ####################################################################################################
    # Generate continuous TTC/OF results 
    with torch.no_grad():
                
        img_list = [] 
        img_list.append(ref_img_big[None, None, :, :, :])
        img_list.append(src_img_big[None, None, :, :, :])
        img_list = torch.cat(img_list, dim=1)
        
        shifts_in = torch.Tensor([[0, 0]]).type(torch.FloatTensor).to(device)

        batch_size = img_list.shape[0]

        # cropping example 
        tw = args.bittcnet_crop_width
        th = args.bittcnet_crop_height
        h = img_list.shape[-2]
        w = img_list.shape[-1]
        # get the top-left corner of the image crop
        xb = np.clip(int(random.randint(0, w - tw) / 3) * 3, 0, max(w - tw - 3, 0)) # top-left corner in the original image
        yb = np.clip(int(random.randint(0, h - th) / 3) * 3, 0, max(h - th - 3, 0)) # top-left corner in the original image
        xcb = xb + (tw - 1) / 2 # get the crop center in the original image 
        ycb = yb + (th - 1) / 2 # get the crop center in the original image 
        xvb = xcb - (w -1) / 2 # get the vector between crop center and original image center 
        yvb = ycb - (h -1) / 2 # get the vector between crop center and original image center 
        xca = xvb / 3 + (w * (args.bittcnet_max_scale) / 3 - 1) / 2 # get the crop center in the feature space 
        yca = yvb / 3 + (h * (args.bittcnet_max_scale) / 3 - 1) / 2 # get the crop center in the feature space 
        xa = int(xca - (tw * (args.bittcnet_max_scale) / 3 - 1) / 2) # get the top-left corner of the cropped feature 
        ya = int(yca - (th * (args.bittcnet_max_scale) / 3 - 1) / 2) # get the top-left corner of the cropped feature 
        
        start_yx_img = torch.Tensor([yb, xb]).type(torch.cuda.LongTensor)[None, :].repeat(batch_size, 1)
        start_yx_fea = torch.Tensor([ya, xa]).type(torch.cuda.LongTensor)[None, :].repeat(batch_size, 1)
                   
        # compute all the image warping transforms 
        vol_params = compute_volume_generation_params(args.attributes, img_list, 
                                               args.alpha_min, args.alpha_max, alpha_delta, 
                                               args.shift_min, args.shift_max, shift_delta)

        seg_ids = vol_params['seg_ids'].repeat(batch_size, 1)

        seg_prob_vol, \
        out_normalized_noisy, \
        out_normalized_final = model(img_list, 
                                     vol_params['T_inv_list_in'], 
                                     vol_params['T_list_in'], 
                                     shifts_in, seg_ids, 
                                     start_yx_fea, start_yx_img) 
        
        print('\nResults generated')

        print('\nVisualizing results')
        
        start_seg = 0
        start_reg = 0

        if 'ttc' in args.attributes:
          out_segt = seg_prob_vol[:, :, :vol_params['eta_size'], ...]
          out_regt = out_normalized_final[:, :1, ...]
          out_regt = out_regt * vol_params['eta_delta'] * vol_params['eta_size'] + vol_params['eta_min_GT']
        
          out_eta_viz = (out_regt[0,0] - 0.5) / (1.0)
          out_eta_viz = disp2rgb(np.clip(out_eta_viz.detach().cpu().numpy(), 0.0, 1.0))
          cv2.imwrite(os.path.join(out_dir, 'eta_out.png'), out_eta_viz[y1:y1+h, x1:x1+w, :] * 255.0)
          start_seg = start_seg + vol_params['eta_size']
          start_reg = start_reg + 1
          

        if 'of' in args.attributes:
          out_segof = seg_prob_vol[:, :, start_seg:, ...]
          out_regof = out_normalized_final[:, start_reg:, ...]
          out_regof[:, 0] = out_regof [:, 0] * vol_params['of_delta'] * vol_params['of_size'] + \
            vol_params['of_min_GT'][0]
          out_regof[:, 1] = out_regof [:, 1] * vol_params['of_delta'] * vol_params['of_size'] + \
            vol_params['of_min_GT'][1]


          out_of_viz = out_regof[0].detach().cpu().numpy().transpose((1, 2, 0))
          v = np.clip(out_of_viz[:, :, 1], args.shift_min, args.shift_max) / (np.sqrt(2) * args.shift_max)
          u = np.clip(out_of_viz[:, :, 0], args.shift_min, args.shift_max) / (np.sqrt(2) * args.shift_max)
          out_of_viz = flow_uv_to_colors(u, v, convert_to_bgr=True)
          cv2.imwrite(os.path.join(out_dir, 'of_out.png'), out_of_viz[y1:y1+h, x1:x1+w, :])

    print('\nDONE!')
        

if __name__ == "__main__":
    main()
