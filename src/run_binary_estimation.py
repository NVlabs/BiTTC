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

from util import disp2rgb, str2bool
from geometry_utils import prepare_warping_transforms, compute_transformed_volume
from geometry_utils import compute_quantized_from_binary_segs, compute_quantized_of_from_binary_segs

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(allow_abbrev=False)

####################################################################################################
# Parse arguments
parser.add_argument('--ref_img_path',                type=str,      default='../data/img_ref.png')
parser.add_argument('--src_img_path',                type=str,      default='../data/img_src.png')
parser.add_argument('--attribute',                   type=str,      default='ttc')

parser.add_argument('--pretrained',                  type=str,      default='../models/binary_ttc_kitti15_train.pth.tar')
parser.add_argument('--arch',                        type=str,      default='bittcnet_binary_of_ttc_2d')
parser.add_argument('--bittcnet_featnet_arch',       type=str,      default='featextractnetspp')
parser.add_argument('--bittcnet_featnethr_arch',     type=str,      default='featextractnethr')
parser.add_argument('--bittcnet_segnet_arch',        type=str,      default='segnet2d')
parser.add_argument('--bittcnet_refinenet_arch',     type=str,      default='segrefinenet')
parser.add_argument('--segnet_num_imgs',             type=int,      default=2)
parser.add_argument('--segnet_num_segs',             type=int,      default=3)
parser.add_argument('--bittcnet_num_refinenets',     type=int,      default=3)
parser.add_argument('--featextractnethr_out_planes', type=int,      default=16)
parser.add_argument('--segrefinenet_in_planes',      type=int,      default=17)
parser.add_argument('--segrefinenet_out_planes',     type=int,      default=8)
parser.add_argument('--segrefinenet_num_layers',     type=int,      default=4)

parser.add_argument('--bittcnet_max_scale',          type=float,    default=1.5)
parser.add_argument('--alpha_vals',                  type=float,    nargs='*',    default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.02, 1.10])
parser.add_argument('--shifts_x',                    type=float,    nargs='*',    default=[-72, -60, -48, -36, -24, -12, 0, 12, 24, 36, 48, 60, 72])
parser.add_argument('--shifts_y',                    type=float,    nargs='*',    default=[-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18])
parser.add_argument('--fps',                         type=float,    default=10)

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

    out_dir = "./results/%s_binary_%s"%(timestamp, args.attribute)
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
    y1 = int(H_full / 2 - h / 2)
    x1 = int(W_full / 2 - w / 2)

    print('\nPadding will be added to ensure input image size divisible by 192')
    print('New input image size is %d x %d'%(H_full, W_full))

    H_out = H_full
    W_out = W_full
    H_in = round(args.bittcnet_max_scale * H_out)
    W_in = round(args.bittcnet_max_scale * W_out)
    H_in = int(H_in)
    W_in = int(W_in)

    options['bittcnet_crop_height'] = H_out
    options['bittcnet_crop_width'] = W_out

    ref_img = torch.Tensor(ref_img).type(torch.float32).permute(2, 0, 1)
    ref_img = transforms.functional.normalize(ref_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ref_img_big = torch.zeros(3, H_full, W_full).type(torch.FloatTensor).cuda()
    ref_img_big[:, int(H_full/2-h/2):int(H_full/2+h/2), int(W_full/2-w/2):int(W_full/2+w/2)] = ref_img

    src_img = torch.Tensor(src_img).type(torch.float32).permute(2, 0, 1)
    src_img = transforms.functional.normalize(src_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    src_img_big = torch.zeros(3, H_full, W_full).type(torch.FloatTensor).cuda()
    src_img_big[:, int(H_full/2-h/2):int(H_full/2+h/2), int(W_full/2-w/2):int(W_full/2+w/2)] = src_img

    ####################################################################################################
    # Prepare ttc and of thresholds 
    if args.attribute == 'ttc':
        print('\n\nComputing binary TTC')
        alpha = np.asarray(args.alpha_vals, dtype=np.float32)
        with np.errstate(divide='ignore'):
            ttc_vals = (1 / args.fps) * np.reciprocal(1 - alpha)
        batch_size = alpha.shape[0]
        print('\nalpha/motion-in-depth threshold values:')
        print(alpha)
        print('\ncorresponding time-to-contact threshould values:')
        print(ttc_vals)
      
        alpha = torch.from_numpy(alpha).to(device)[:, None]
        theta = alpha - 1.0
    
    elif args.attribute == 'of':
        print('\n\nComputing 2D binary OF')
        shifts_x = np.asarray(args.shifts_x, dtype=np.float32)
        shifts_y = np.asarray(args.shifts_y, dtype=np.float32)
        if not (shifts_x.shape[0] == shifts_y.shape[0]):
           print('\nThis code computes binary segmentation along x and y at the same time.')
           print('shifts_x and shifts_y should have same size.')
           exit(-1)
        
        print('\n2D Binary optical flow segmentation will be computed with respect to following thresholds:')
        for i in range (shifts_x.shape[0]): print('(%.2f, %.2f)'%(shifts_x[i], shifts_y[i]))
        
        shifts_x = torch.from_numpy(shifts_x).to(device)[:, None]
        shifts_y = torch.from_numpy(shifts_y).to(device)[:, None]
        batch_size = shifts_x.shape[0]
        
        
    ####################################################################################################
    # MODEL
    network_data = torch.load(args.pretrained)
    print("\n=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](options, network_data).cuda().eval()

    ####################################################################################################
    # prepare tensors for some pretty visualizations
    K_s, K_d, T_gs_s, T_gs_d, \
    grid_template_in, grid_template_out = prepare_warping_transforms(H_in, W_in, H_out, W_out, torch.float32)
    K_s = K_s.to(device) 
    K_d = K_d.to(device) 
    K_s_inv = torch.inverse(K_s)
    K_d_inv = torch.inverse(K_d)
    T_gs_s = T_gs_s.to(device) 
    T_gs_d = T_gs_d.to(device) 
    grid_template_in = grid_template_in.to(device) 
    grid_template_out = grid_template_out.to(device) 

    ####################################################################################################
    # Generate binary and quantized results 
    with torch.no_grad():
        
        img_list = [] 
        img_list.append(ref_img_big[None, None, :, :, :])
        img_list.append(src_img_big[None, None, :, :, :])
        img_list = torch.cat(img_list, dim=1)
        img_list = img_list.repeat(batch_size, 1, 1, 1, 1)

        if args.attribute == 'ttc':
            alpha_for = torch.mul(torch.arange(0, 2).type_as(alpha)[None, :], theta) + 1.0
            alpha_inv = torch.reciprocal(alpha_for)
            is_compute_TTC = True
        else:
            alpha_for = torch.ones(batch_size, 2).type_as(img_list)
            alpha_inv = torch.reciprocal(alpha_for)
            is_compute_TTC = False
        
        if args.attribute == 'of':
            delta_list = torch.arange(0, 2).type_as(shifts_x)[None, :]
            
            shiftx_inv = torch.mul(delta_list, shifts_x)
            shiftx_for =-torch.mul(delta_list, shifts_x)    
            segx_thr = shifts_x.repeat(1, 2)
            
            shifty_inv = torch.mul(delta_list, shifts_y)
            shifty_for =-torch.mul(delta_list, shifts_y)
            segy_thr = shifts_y.repeat(1, 2)

            shiftof_for = torch.cat([shiftx_for[None, :, :], shifty_for[None, :, :]], dim=0)
            shiftof_inv = torch.cat([shiftx_inv[None, :, :], shifty_inv[None, :, :]], dim=0)
            segof_thr   = torch.cat([segx_thr[None, :, :], segy_thr[None, :, :]], dim=0)
        else:
            shiftof_for = torch.zeros(2, batch_size, 2).type_as(img_list)
            shiftof_inv = torch.zeros(2, batch_size, 2).type_as(img_list)
        
        # compute transform
        T_inv_list = torch.eye(3)[None, None, None, :, :].type_as(img_list)
        T_inv_list = T_inv_list.repeat(batch_size, 2, 1, 1, 1)
        T_inv_list[:, :, 0, 0, 2] = shiftof_inv[0] / 3 # since feature maps are 3x downsampled
        T_inv_list[:, :, 0, 1, 2] = shiftof_inv[1] / 3 # since feature maps are 3x downsampled
        T_inv_list[:, :, 0, 0, 0] = alpha_inv
        T_inv_list[:, :, 0, 1, 1] = alpha_inv
        T_list = torch.eye(3)[None, None, None, :, :].type_as(img_list)
        T_list = T_list.repeat(batch_size, 2, 1, 1, 1)
        T_list[:, :, 0, 0, 2] = shiftof_for[0] / 3 # since feature maps are 3x downsampled
        T_list[:, :, 0, 1, 2] = shiftof_for[1] / 3 # since feature maps are 3x downsampled
        T_list[:, :, 0, 0, 0] = alpha_for
        T_list[:, :, 0, 1, 1] = alpha_for
        
        # forward pass 
        out = model(img_list, T_inv_list, T_list, is_compute_TTC)[1]

        print('\nResults generated')

        print('\nVisualizing results')
        
        # generate visualizations 
        if args.attribute == 'ttc':
            T_inv_list_t = T_inv_list.clone()
            input_list_in = [img_list.clone()]
            
            # generate warped images that implicitly pose a binary task to the network
            input_list_in = compute_transformed_volume(input_list_in,
                                                       T_inv_list_t, H_in, W_in,
                                                       K_s, K_d_inv, T_gs_s,
                                                       grid_template_in)
            
            mean_values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
            std_values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
            quant_vals = alpha[:, 0].cpu().numpy()

            for j in range(input_list_in[0].shape[0]):
                for d in range(input_list_in[0].shape[1]):
                    img_viz = (input_list_in[0][j,d].cpu().mul_(std_values) + mean_values).clamp(0, 1)
                    img_path = os.path.join(out_dir, '%03d_thr%.2f_img_in_%d.jpg'%(j, quant_vals[j], d))
                    cv2.imwrite(img_path, np.transpose(img_viz.numpy() * 255.0, (1, 2, 0)))

            # visualize binary segmentation probability maps
            out_segt = out[:, :, 0:1, :, :].clone()
            segs = []
            for d in range(out_segt.shape[0]):
                out_segt_viz = out_segt[d, 0, :, :, :].detach().clone().cpu().numpy()
                segs.append(out_segt_viz.copy())
                cv2.imwrite(os.path.join(out_dir, '%03d_thr%.2f_seg_out.jpg'%
                    (d, quant_vals[d])), 255.0 - out_segt_viz[0, y1:y1+h, x1:x1+w] * 255.0)
            
            # visualize quantized output 
            segs = np.concatenate(segs, axis=0)
            
            if (np.sum(quant_vals[1:]-quant_vals[:-1] < 0) == 0):
              print('\nalpha threshold values are in increasing order')
              print('Computing quantized TTC map')  
              label_map, quant_map, quant_map_rgb = compute_quantized_from_binary_segs(segs, quant_vals)
              cv2.imwrite(os.path.join(out_dir,'quant_ttc_out.jpg'), quant_map_rgb[y1:y1+h, x1:x1+w, :])

                
        if args.attribute == 'of':
            T_inv_list_of = T_inv_list.clone()
            T_inv_list_of[:, :, 0, 0, 2] *= 3 # Since we are performing shift on images instead of features
            T_inv_list_of[:, :, 0, 1, 2] *= 3 # Since we are performing shift on images instead of features
            
            input_list_in = [img_list.clone()]
            
            input_list_in = compute_transformed_volume(input_list_in,
                                                       T_inv_list_of, H_in, W_in,
                                                       K_s, K_d_inv, T_gs_s,
                                                       grid_template_in)
            
            mean_values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
            std_values  = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
            quantx_vals = shifts_x[:, 0].cpu().numpy()
            quanty_vals = shifts_y[:, 0].cpu().numpy()

            for j in range(input_list_in[0].shape[0]):
                for d in range(input_list_in[0].shape[1]):
                    img_viz = (input_list_in[0][j, d].cpu().mul_(std_values) + mean_values).clamp(0, 1)
                    img_path = os.path.join(out_dir, '%03d_thrx%.2f_thry%.2f_img_in_%03d.jpg'%
                        (j, quantx_vals[j], quanty_vals[j], d))
                    cv2.imwrite(img_path, np.transpose(img_viz.numpy()*255.0, (1, 2, 0)))

            out_segof = out[:, :, -2:, :, :].clone()

            segs_x = []
            segs_y = []
            for d in range(out_segof.shape[0]):
                out_segof_viz = np.zeros((out_segof.shape[-2], out_segof.shape[-1], 3), dtype=np.float32)
                out_segof_viz[:, :, 1] = 1.0 - out_segof[d, 0, 0, :, :].detach().clone().cpu().numpy()
                out_segof_viz[:, :, 2] = 1.0 - out_segof[d, 0, 1, :, :].detach().clone().cpu().numpy()

                segs_x.append(out_segof[d, 0, 0:1, :, :].detach().clone().cpu().numpy())
                segs_y.append(out_segof[d, 0, 1:2, :, :].detach().clone().cpu().numpy())
                cv2.imwrite(os.path.join(out_dir, '%03d_thrx%.2f_thry%.2f_seg_out.jpg'%
                    (d, quantx_vals[d], quanty_vals[d])), out_segof_viz[y1:y1+h, x1:x1+w, :] * 255.0)
            
            segs_x = np.concatenate(segs_x, axis=0)
            if (np.sum(quantx_vals[1:]-quantx_vals[:-1] < 0) == 0):
                print('\nshifts_x threshold values are in increasing order')
                print('Computing quantized optical flow in x direction')  
                label_map, quant_map, quant_map_rgb = compute_quantized_of_from_binary_segs(segs_x, quantx_vals)
                cv2.imwrite(os.path.join(out_dir, 'quant_ofx_out.jpg'), quant_map_rgb[y1:y1+h, x1:x1+w, :])

            segs_y = np.concatenate(segs_y, axis=0)
            if (np.sum(quanty_vals[1:]-quanty_vals[:-1] < 0) == 0):
                print('\nshifts_y threshold values are in increasing order')
                print('Computing quantized optical flow in y direction')  
                label_map, quant_map, quant_map_rgb = compute_quantized_of_from_binary_segs(segs_y, quanty_vals)
                cv2.imwrite(os.path.join(out_dir, 'quant_ofy_out.jpg'), quant_map_rgb[y1:y1+h, x1:x1+w, :])
    
    print('\nDONE!')
        

if __name__ == "__main__":
    main()
