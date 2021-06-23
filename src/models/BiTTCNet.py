# Copyright 2021 NVIDIA CORPORATION & AFFILIATES
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from   torch.autograd import Variable
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import sys
sys.path.append('./')

import models.FeatExtractNet as FeatNet
import models.SegNet2D as SegNet2D
import models.RefineNet2D as RefineNet2D


from geometry_utils import prepare_warping_transforms
from geometry_utils import compute_transformed_volume
from geometry_utils import remove_transformed_volume_padding

__all__ = ['bittcnet_binary_of_ttc_2d','bittcnet_continuous_of_ttc_2d']


# Binary TTC and OF segmentation network
class BiTTCNetBinaryOFTTC2D(nn.Module):
    def __init__(self, options, featnet_arch, segnet_arch, 
                 featnethr_arch=None, refinenet_arch=None, 
                 num_refinenets=1, H=None, W=None, max_scale=None):

        super(BiTTCNetBinaryOFTTC2D, self).__init__()

        self.max_scale = max_scale

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False
        
        self.featnet = FeatNet .__dict__[featnet_arch](options, data=None)
        self.segnet2D = SegNet2D.__dict__[segnet_arch] (options, data=None)
        if self.is_refine:
            self.featnethr = FeatNet.__dict__[featnethr_arch](options, data=None)
            self.refinenets = []
            for i in range(num_refinenets):
                self.refinenets.append(RefineNet2D.__dict__[refinenet_arch](options, data=None))
            self.refinenets = nn.ModuleList(self.refinenets)
                
        # H and W are the original input image size, get the feature sizes by dividing by 3
        self.H_f_out = int(H / 3)
        self.W_f_out = int(W / 3)
        self.H_f_in = int(round(self.max_scale * self.H_f_out))
        self.W_f_in = int(round(self.max_scale * self.W_f_out))

        # prepare warping transforms
        K_s, K_d, T_gs_s, T_gs_d, \
        grid_template_in, grid_template_out = \
            prepare_warping_transforms(self.H_f_in, self.W_f_in, self.H_f_out, self.W_f_out, dtype=options['dtype'])
        K_s_inv = torch.inverse(K_s)
        K_d_inv = torch.inverse(K_d)
        self.register_buffer('K_s', K_s, persistent=False)
        self.register_buffer('K_d', K_d, persistent=False)
        self.register_buffer('K_s_inv', K_s_inv, persistent=False)
        self.register_buffer('K_d_inv', K_d_inv, persistent=False)
        self.register_buffer('T_gs_s', T_gs_s, persistent=False)
        self.register_buffer('T_gs_d', T_gs_d, persistent=False)
        self.register_buffer('grid_template_in', grid_template_in, persistent=False)
        self.register_buffer('grid_template_out', grid_template_out, persistent=False)

        return
       
    def forward(self, img_list, T_inv_list, T_list, is_compute_TTC):

        batch_size = img_list.shape[0]
        dolly_size = img_list.shape[1]
        H_out = img_list.shape[3]
        W_out = img_list.shape[4]
                
        # feature extraction 
        features_list = self.featnet(img_list.reshape(batch_size * dolly_size, 3, H_out, W_out))
        W_f_out = features_list.shape[-1]
        H_f_out = features_list.shape[-2]
        num_F = features_list.shape[-3]
        features_list = features_list.reshape(batch_size, dolly_size, num_F, H_f_out, W_f_out)

        if self.is_refine:
            features_refhr = self.featnethr(img_list[:, 0, :, :, :])

        # warped volume generation
        [dzv] = compute_transformed_volume([features_list], T_inv_list, 
                                            self.H_f_in, self.W_f_in,
                                            self.K_s, self.K_d_inv, self.T_gs_s, 
                                            self.grid_template_in)

        dzv = dzv.reshape(dzv.shape[0], dolly_size * num_F, self.H_f_in, self.W_f_in) # concat 
        
        # segmentation using 2D network
        seg_raw_low_res = self.segnet2D(dzv)

        # remove padding 
        seg_raw_low_res = seg_raw_low_res.reshape(seg_raw_low_res.shape[0], 1, 
                                                  seg_raw_low_res.shape[1], 
                                                  self.H_f_in, self.W_f_in) 
        [seg_raw_low_res] = \
            remove_transformed_volume_padding([seg_raw_low_res], 
                                              H_f_out, W_f_out,
                                              self.K_s_inv, self.K_d, self.T_gs_d, 
                                              self.grid_template_out, 
                                              T_list[:, 0, :, :, :][:, None, :, :, :])
        
        # generate unrefined segmentation
        seg_raw_low_res = seg_raw_low_res[:, 0, :, :, :]
        seg_prob_low_res = torch.sigmoid(seg_raw_low_res)
        seg_prob_low_res_up = F.interpolate(seg_prob_low_res, 
                                            size = [H_out, W_out],
                                            mode = 'bilinear',
                                            align_corners = False)
        out=[]
        out.append(seg_prob_low_res_up[:, None, :, :, :])

        # generate refined result
        if self.is_refine:
            seg_raw_low_res_up = F.interpolate(seg_raw_low_res, 
                                               size=[H_out, W_out],
                                               mode='bilinear',
                                               align_corners=False)
            # Refine Net
            seg_raw_high_res = []
            if is_compute_TTC:
                refine_net_input_0 = torch.cat((seg_raw_low_res_up[:, 0, :, :][:, None, :, :], 
                                                features_refhr), dim=1)
                seg_raw_high_res.append(self.refinenets[0](refine_net_input_0))
            else:
                refine_net_input_1 = torch.cat((seg_raw_low_res_up[:, 1, :, :][:, None, :, :], 
                                                features_refhr), dim=1)
                seg_raw_high_res.append(self.refinenets[1](refine_net_input_1))
            
                refine_net_input_2 = torch.cat((seg_raw_low_res_up[:, 2, :, :][:, None, :, :], 
                                                features_refhr), dim=1)
                seg_raw_high_res.append(self.refinenets[2](refine_net_input_2))

            seg_raw_high_res  = torch.cat(seg_raw_high_res, dim=1)
            seg_prob_high_res = torch.sigmoid(seg_raw_high_res)
            out.append(seg_prob_high_res[:, None, :, :, :])
        else:
            out.append(seg_prob_low_res_up[:, None, :, :, :])
        
        return out

    
def bittcnet_binary_of_ttc_2d(options, data=None):
    print('==> USING BiTTCNetBinaryOFTTC2D')
    for key in options:
        if 'bittcnet' in key:
            print('{} : {}'.format(key, options[key]))

    
    model = BiTTCNetBinaryOFTTC2D(options,
                                  featnet_arch = options['bittcnet_featnet_arch'],
                                  segnet_arch = options['bittcnet_segnet_arch'],
                                  featnethr_arch = options['bittcnet_featnethr_arch'],
                                  refinenet_arch = options['bittcnet_refinenet_arch'],
                                  num_refinenets = options['bittcnet_num_refinenets'],
                                  H = options['bittcnet_crop_height'], 
                                  W = options['bittcnet_crop_width'], 
                                  max_scale = options['bittcnet_max_scale'])

    if data is not None:
        model.load_state_dict(data['state_dict'])
        
    return model


# Continuous TTC and OF via binary segmentation network
class BiTTCNetContinuousOFTTC2D(nn.Module):
    def __init__(self, options, featnet_arch, segnet_arch, 
                 H_C=None, W_C=None, H_F=None, W_F=None, max_scale=None):

        super(BiTTCNetContinuousOFTTC2D, self).__init__()

        self.max_scale = max_scale

        # core segmentation network  
        self.featnet = FeatNet .__dict__[featnet_arch](options, data=None)
        self.segnet2D = SegNet2D.__dict__[segnet_arch](options, data=None)
        
        # refinement network
        self.refinenet = RefineNet2D.__dict__['regrefinenet'](options, data=None)
        
        # supporting variables
        # H_F and W_F are the original input image size
        self.H_F_f_out = int(H_F / 3)
        self.W_F_f_out = int(W_F / 3)
        self.H_F_f_in = int(self.max_scale * self.H_F_f_out) # padding to allow warping without cropping
        self.W_F_f_in = int(self.max_scale * self.W_F_f_out) # padding to allow warping without cropping
        # H_C and W_C are cropped image size
        self.H_C = H_C
        self.W_C = W_C
        self.H_C_f_out = int(H_C / 3)
        self.W_C_f_out = int(W_C / 3)
        self.H_C_f_in = int(self.max_scale * self.H_C_f_out) # padding to allow warping without cropping
        self.W_C_f_in = int(self.max_scale * self.W_C_f_out) # padding to allow warping without cropping

        # prepare warping transforms 
        K_s, K_d, T_gs_s, T_gs_d, \
        grid_template_in, grid_template_out = \
            prepare_warping_transforms(self.H_F_f_in, self.W_F_f_in, self.H_F_f_out, self.W_F_f_out, 
                                       dtype=options['dtype'])
        K_d_inv = torch.inverse(K_d)
        self.register_buffer('K_s', K_s, persistent=False)
        self.register_buffer('K_d_inv', K_d_inv, persistent=False)
        self.register_buffer('T_gs_s', T_gs_s, persistent=False)
        self.register_buffer('grid_template_in', grid_template_in, persistent=False)

        K_s_c, K_d_c, T_gs_s_c, T_gs_d_c, \
        grid_template_in_c, grid_template_out_c = \
            prepare_warping_transforms(self.H_C_f_in, self.W_C_f_in, self.H_C_f_out, self.W_C_f_out, 
                                       dtype=options['dtype'])
        K_s_inv = torch.inverse(K_s_c)
        self.register_buffer('K_s_inv', K_s_inv, persistent=False)
        self.register_buffer('K_d', K_d_c, persistent=False)
        self.register_buffer('T_gs_d', T_gs_d_c, persistent=False)
        self.register_buffer('grid_template_out', grid_template_out_c, persistent=False)

        return
       
    def forward(self, img_list, T_inv_lists, T_lists, shifts, seg_ids, start_yx_fea, start_yx_img):

        seg_prob_vol = []
        out_normalized_noisy = []
        out_normalized_final = []
        # perform inference on different volumes for TTC, OF_X, OF_Y
        for i in range(len(T_inv_lists)):
            seg_prob_vol_i, \
            out_normalized_noisy_i, \
            out_normalized_final_i = self.forward_per_vol(img_list,
                                                          T_inv_lists[i],
                                                          T_lists[i],
                                                          shifts,
                                                          seg_ids[:, i:i+1], 
                                                          start_yx_fea, start_yx_img)
            seg_prob_vol.append(seg_prob_vol_i)
            out_normalized_noisy.append(out_normalized_noisy_i)
            out_normalized_final.append(out_normalized_final_i)

        # concatenate everything 
        seg_prob_vol = torch.cat(seg_prob_vol, dim=2)
        out_normalized_noisy = torch.cat(out_normalized_noisy, dim=1)
        out_normalized_final = torch.cat(out_normalized_final, dim=1)

        return seg_prob_vol, out_normalized_noisy, out_normalized_final

    def forward_per_vol(self, img_list, T_inv_list, T_list, shifts, seg_ids, start_yx_fea, start_yx_img):

        batch_size = img_list.shape[0]
        dolly_size = img_list.shape[1]
        H_F_out = img_list.shape[3]
        W_F_out = img_list.shape[4]

        # Feature Extraction 
        features_list  = self.featnet(img_list.reshape(batch_size*dolly_size, 3, H_F_out, W_F_out))
        W_f_out = features_list.shape[-1]
        H_f_out = features_list.shape[-2]
        num_F = features_list.shape[-3]
        features_list = features_list.reshape(batch_size, dolly_size, num_F, H_f_out, W_f_out)

        total_vol_size = T_inv_list.shape[1]
        
        features_list_rep = features_list[:, None, :, :, :, :].repeat(1, total_vol_size, 1, 1, 1, 1)
        features_list_rep_in = features_list_rep.view(-1, dolly_size, num_F, H_f_out, W_f_out)

        shifts_rep = shifts[:, None, :].repeat(1, T_inv_list.shape[1], 1).view(-1, 2)

        [dzv] = compute_transformed_volume([features_list_rep_in], 
                                           T_inv_list.view(-1, dolly_size, 1, 3, 3),
                                           self.H_F_f_in, self.W_F_f_in,
                                           self.K_s, self.K_d_inv, self.T_gs_s,
                                           self.grid_template_in, shifts_rep)

        # Segmentation Network 2D
        dzv = dzv.reshape(batch_size, total_vol_size, dolly_size, num_F, self.H_F_f_in, self.W_F_f_in) # concat 
        dzv = dzv.reshape(batch_size, total_vol_size, dolly_size * num_F, self.H_F_f_in, self.W_F_f_in) # concat 
        dzv = dzv.reshape(batch_size * total_vol_size, dolly_size * num_F, self.H_F_f_in, self.W_F_f_in) # concat 

        # crop dzv 
        dzv = dzv[..., start_yx_fea[0,0]:start_yx_fea[0,0]+self.H_C_f_in, start_yx_fea[0,1]:start_yx_fea[0,1]+self.W_C_f_in]
        
        seg_raw_low_res = self.segnet2D(dzv)
            
        seg_raw_low_res = seg_raw_low_res.view(seg_raw_low_res.shape[0], 1, 
                                               seg_raw_low_res.shape[1], 
                                               self.H_C_f_in, self.W_C_f_in)        
        [seg_raw_low_res] = remove_transformed_volume_padding([seg_raw_low_res], 
                                                              self.H_C_f_out, self.W_C_f_out,
                                                              self.K_s_inv, self.K_d, self.T_gs_d, 
                                                              self.grid_template_out, 
                                                              T_list.view(-1, 1, 1, 3, 3))
        seg_raw_low_res = seg_raw_low_res.view(batch_size, 
                                               total_vol_size, 
                                               seg_raw_low_res.shape[-4], 
                                               seg_raw_low_res.shape[-3], 
                                               seg_raw_low_res.shape[-2], 
                                               seg_raw_low_res.shape[-1])
        
        seg_raw_low_res = seg_raw_low_res[:, :, 0, :, :, :] # remove dolly axis as we only need reference   
        
        seg_raw_low_res = seg_raw_low_res[:, None, :, seg_ids[0,0], :, :]
    
        seg_prob_low_res_up = torch.sigmoid(F.interpolate(seg_raw_low_res, 
                                                          size = [total_vol_size * 3, self.H_C, self.W_C],
                                                          mode = 'trilinear',
                                                          align_corners = False))
        seg_prob_low_res_up = seg_prob_low_res_up[:, :, 1:-1, :, :]
            
        out_normalized_noisy = torch.mean(seg_prob_low_res_up, dim=2, keepdim=False)
        img_crop = img_list[:, 0, :, 
                            start_yx_img[0, 0]:start_yx_img[0, 0] + self.H_C,
                            start_yx_img[0, 1]:start_yx_img[0, 1] + self.W_C]
        refinenet_input = torch.cat((out_normalized_noisy, img_crop), dim=1)
        
        out_normalized_final = self.refinenet(refinenet_input)
        
        return seg_prob_low_res_up, out_normalized_noisy, out_normalized_final

    
def bittcnet_continuous_of_ttc_2d(options, data=None):
    print('==> USING BiTTCNetContinuousOFTTC2D')
    for key in options:
        if 'bittcnet' in key:
            print('{} : {}'.format(key, options[key]))

    
    model = BiTTCNetContinuousOFTTC2D(options,
                                      featnet_arch = options['bittcnet_featnet_arch'],
                                      segnet_arch = options['bittcnet_segnet_arch'],
                                      H_C = options['bittcnet_crop_height'], 
                                      W_C = options['bittcnet_crop_width'], 
                                      H_F = options['bittcnet_full_height'], 
                                      W_F = options['bittcnet_full_width'], 
                                      max_scale = options['bittcnet_max_scale'])

    if data is not None:
        model.load_state_dict(data['state_dict'])
        
    return model
