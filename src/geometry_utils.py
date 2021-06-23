# Copyright 2021 NVIDIA CORPORATION & AFFILIATES
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np 
import torch 
import torch.nn.functional as F
from util import disp2rgb


def compute_quantized_from_binary_segs(segs, quant_vals):
    h = segs[0].shape[-2]
    w = segs[0].shape[-1]
    segs = np.insert(segs, 0, np.ones((1, h, w), dtype=np.float32), axis=0)
    segs = np.append(segs, np.zeros((1, h, w), dtype=np.float32), axis=0)
    segs = 1.0 - segs
    pdfs = segs[1:, :, :] - segs[:-1, :, :]
    pdfs[pdfs<0] = 0
    labels_method = np.argmax(pdfs, axis=0).astype(np.int)
    quant_map = labels_method.astype(np.float32)
    quant_vals = np.insert(quant_vals, 0, min(quant_vals[0], 0.5))
    quant_vals = np.append(quant_vals, max(quant_vals[-1], 1.5))

    for i in range(len(quant_vals) - 1):
        min_quant_val = quant_vals[i]
        max_quant_val = quant_vals[i + 1]
        mid_quant_val = 0.5 * (min_quant_val + max_quant_val)
        quant_map[labels_method == i] = mid_quant_val

    quant_map_rgb = np.clip(quant_map, quant_vals[0], quant_vals[-1])    
    quant_map_rgb = (quant_map_rgb - quant_vals[0]) / (quant_vals[-1] - quant_vals[0])
    quant_map_rgb = (disp2rgb(quant_map_rgb) * 255.0).astype(np.uint8)

    return labels_method, quant_map, quant_map_rgb


def compute_quantized_of_from_binary_segs(segs, quant_vals):
    h = segs[0].shape[-2]
    w = segs[0].shape[-1]
    segs = np.insert(segs, 0, np.ones((1, h, w), dtype=np.float32), axis=0)
    segs = np.append(segs, np.zeros((1, h, w), dtype=np.float32), axis=0)
    segs = 1.0 - segs
    pdfs = segs[1:, :, :] - segs[:-1, :, :]
    pdfs[pdfs<0] = 0
    labels_method = np.argmax(pdfs, axis=0).astype(np.int)
    quant_map = labels_method.astype(np.float32)
    quant_vals = np.insert(quant_vals, 0, min(quant_vals[0], -96))
    quant_vals = np.append(quant_vals, max(quant_vals[-1], 96))

    for i in range(len(quant_vals) - 1):
        min_quant_val = quant_vals[i]
        max_quant_val = quant_vals[i + 1]
        mid_quant_val = 0.5 * (min_quant_val + max_quant_val)
        quant_map[labels_method == i] = mid_quant_val

    quant_map_rgb = np.clip(quant_map, quant_vals[0], quant_vals[-1])    
    quant_map_rgb = (quant_map_rgb - quant_vals[0]) / (quant_vals[-1] - quant_vals[0])
    quant_map_rgb = (disp2rgb(quant_map_rgb) * 255.0).astype(np.uint8)

    return labels_method, quant_map, quant_map_rgb


def prepare_warping_transforms(H_in, W_in, H_out, W_out, dtype):

    K_s = np.asarray([[1.0, 0.0, W_out / 2.0 - 0.5],
                      [0.0, 1.0, H_out / 2.0 - 0.5],
                      [0.0, 0.0, 1.0              ]], dtype=np.float32)
    K_d = np.asarray([[1.0, 0.0, W_in / 2.0 - 0.5 ],
                      [0.0, 1.0, H_in / 2.0 - 0.5 ],
                      [0.0, 0.0, 1.0              ]], dtype=np.float32)
    
    T_gs_s = np.asarray([[2.0 / W_out, 0.0        , 1.0 / W_out - 1.0],
                         [0.0        , 2.0 / H_out, 1.0 / H_out - 1.0],
                         [0.0        , 0.0        , 1.0              ]], dtype=np.float32)
    T_gs_d = np.asarray([[2.0 / W_in, 0.0       , 1.0 / W_in - 1.0],
                         [0.0       , 2.0 / H_in, 1.0 / H_in - 1.0],
                         [0.0       , 0.0       , 1.0             ]], dtype=np.float32)

    K_s = torch.from_numpy(K_s).type(dtype=dtype)
    K_d = torch.from_numpy(K_d).type(dtype=dtype)
    T_gs_s = torch.from_numpy(T_gs_s).type(dtype=dtype)
    T_gs_d = torch.from_numpy(T_gs_d).type(dtype=dtype)

    # prepare the grid for sampling 
    xx = torch.arange(0, W_in).view(1, -1).repeat(H_in, 1)
    yy = torch.arange(0, H_in).view(-1, 1).repeat(1, W_in)
    xx = xx.view(1, 1, H_in, W_in)
    yy = yy.view(1, 1, H_in, W_in)
    zz = torch.ones_like(xx)
    grid_template_in = torch.cat((xx, yy, zz), 1).type(dtype=dtype)
    grid_template_in = grid_template_in.permute((0, 2, 3, 1)).view(1, -1, 3).unsqueeze(0).unsqueeze(-1)
    grid_template_in = grid_template_in.contiguous()

    # prepare the grid for sampling 
    xx = torch.arange(0, W_out).view(1, -1).repeat(H_out, 1)
    yy = torch.arange(0, H_out).view(-1, 1).repeat(1, W_out)
    xx = xx.view(1, 1, H_out, W_out)
    yy = yy.view(1, 1, H_out, W_out)
    zz = torch.ones_like(xx)
    grid_template_out = torch.cat((xx, yy, zz), 1).type(dtype=dtype)
    grid_template_out = grid_template_out.permute((0, 2, 3, 1)).view(1, -1, 3).unsqueeze(0).unsqueeze(-1)
    grid_template_out = grid_template_out.contiguous()

    return K_s, K_d, T_gs_s, T_gs_d, grid_template_in, grid_template_out


def compute_transformed_volume(input_sequence_list, T_inv_list, H_in, W_in,
                         K_s, K_d_inv, T_gs_s, grid_template_in, shifts=None):

    batch_size = input_sequence_list[0].shape[0]
    dolly_size = input_sequence_list[0].shape[1]
    H_out = input_sequence_list[0].shape[3]
    W_out = input_sequence_list[0].shape[4]

    if torch.is_tensor(shifts):
        shifts_repeat = shifts[:, None, :].repeat(1, dolly_size, 1)
        T_shift = torch.eye(3)[None, None, None, :, :].type_as(shifts)
        T_shift = T_shift.repeat(batch_size, dolly_size, 1, 1, 1)
        T_shift[:, :, 0, 0, 2] = shifts_repeat[:, :, 0]
        T_shift[:, :, 0, 1, 2] = shifts_repeat[:, :, 1]
        T_shift_inv = torch.eye(3)[None, None, None, :, :].type_as(shifts)
        T_shift_inv = T_shift_inv.repeat(batch_size, dolly_size, 1, 1, 1)
        T_shift_inv[:, :, 0, 0, 2] = -shifts_repeat[:, :, 0]
        T_shift_inv[:, :, 0, 1, 2] = -shifts_repeat[:, :, 1]
        T_inv_list = torch.matmul(T_shift_inv, torch.matmul(T_inv_list, T_shift))

    grid = grid_template_in.repeat(batch_size, dolly_size, 1, 1, 1)
    T_img_in = torch.matmul(T_gs_s, torch.matmul(K_s, torch.matmul(T_inv_list, K_d_inv)))
    grid_img_in = torch.matmul(T_img_in, grid).squeeze(-1)
    grid_img_in = grid_img_in.reshape(batch_size, dolly_size, H_in, W_in, 3)[:, :, :, :, :2]

    output_sequence_list = []
    for input_sequence in input_sequence_list:
        num_F = input_sequence.shape[2]
        output_sequence = F.grid_sample(
                            input_sequence.reshape(-1, num_F, H_out, W_out), 
                            grid_img_in.reshape(-1, H_in, W_in, 2), 
                            mode='bilinear', 
                            padding_mode='zeros')
        output_sequence = output_sequence.reshape(batch_size, dolly_size, num_F, H_in, W_in)
        output_sequence_list.append(output_sequence)
        
    return output_sequence_list



def remove_transformed_volume_padding(input_sequence_list, H_out, W_out, 
                                      K_s_inv, K_d, T_gs_d, grid_template_out, T_list):

    batch_size = input_sequence_list[0].shape[0]
    dolly_size = input_sequence_list[0].shape[1]
    H_in = input_sequence_list[0].shape[3]
    W_in = input_sequence_list[0].shape[4]
    
    grid = grid_template_out.repeat(batch_size, dolly_size, 1, 1, 1)
    T_img_out = torch.matmul(T_gs_d, torch.matmul(K_d, torch.matmul(T_list, K_s_inv)))
    grid_img_out = torch.matmul(T_img_out, grid).squeeze(-1)
    grid_img_out = grid_img_out.reshape(batch_size, dolly_size, H_out, W_out, 3)[:, :, :, :, :2]

    output_sequence_list = []
    for input_sequence in input_sequence_list:
        num_F = input_sequence.shape[2]
        output_sequence = F.grid_sample(input_sequence.reshape(-1, num_F, H_in, W_in), 
                                        grid_img_out.reshape(-1, H_out, W_out, 2), 
                                        mode='bilinear', 
                                        padding_mode='zeros')
        output_sequence = output_sequence.reshape(batch_size, dolly_size, num_F, H_out, W_out)
        output_sequence_list.append(output_sequence)

    return output_sequence_list



def compute_params_of_vol(img_list, shiftx_list, shifty_list):

    batch_size = img_list.shape[0]
    dolly_size = img_list.shape[1]
    vol_size = shiftx_list.shape[0]
    H_out = img_list.shape[-2]
    W_out = img_list.shape[-1]

    delta_list = torch.arange(0, dolly_size).type_as(img_list)[None, :]
    
    shiftx = shiftx_list
    shiftx_inv = torch.mul(delta_list, shiftx)
    shiftx_for =-torch.mul(delta_list, shiftx)    
    
    T_ofx_inv_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_ofx_inv_list = T_ofx_inv_list.repeat(1, vol_size, dolly_size, 1, 1, 1)
    T_ofx_inv_list[0, :, :, 0, 0, 2] = shiftx_inv
    T_ofx_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_ofx_list = T_ofx_list.repeat(1, vol_size, dolly_size, 1, 1, 1)
    T_ofx_list[0, :, :, 0, 0, 2] = shiftx_for
    
    
    shifty = shifty_list
    shifty_inv = torch.mul(delta_list, shifty)
    shifty_for =-torch.mul(delta_list, shifty)
    
    T_ofy_inv_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_ofy_inv_list = T_ofy_inv_list.repeat(1, vol_size, dolly_size, 1, 1, 1)
    T_ofy_inv_list[0, :, :, 0, 1, 2] = shifty_inv
    T_ofy_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_ofy_list = T_ofy_list.repeat(1, vol_size, dolly_size, 1, 1, 1)
    T_ofy_list[0, :, :, 0, 1, 2] = shifty_for


    # add repeat across batch_size 
    T_ofx_inv_list = T_ofx_inv_list.repeat(batch_size, 1, 1, 1, 1, 1)
    T_ofx_list = T_ofx_list.repeat(batch_size, 1, 1, 1, 1, 1)

    T_ofy_inv_list = T_ofy_inv_list.repeat(batch_size, 1, 1, 1, 1, 1)
    T_ofy_list = T_ofy_list.repeat(batch_size, 1, 1, 1, 1, 1)
    
    # prepare targets

    T_ofx_list = T_ofx_list[:, :, 0:1, :, :, :]
    T_ofy_list = T_ofy_list[:, :, 0:1, :, :, :]

    out = {'T_ofx_list':T_ofx_list,
           'T_ofy_list':T_ofy_list,
           'T_ofx_inv_list':T_ofx_inv_list,
           'T_ofy_inv_list':T_ofy_inv_list}

    return out


def compute_params_ttc_vol(img_list, alpha_list):

    batch_size = img_list.shape[0]
    vol_size = alpha_list.shape[0]

    alpha = torch.mul(torch.arange(0, 2).type_as(img_list)[None, :], alpha_list - 1) + 1

    alpha_for = alpha 
    alpha_inv = torch.reciprocal(alpha_for)

    T_inv_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_inv_list = T_inv_list.repeat(1, vol_size, 2, 1, 1, 1)
    T_inv_list[0, :, :, 0, 0, 0] = alpha_inv
    T_inv_list[0, :, :, 0, 1, 1] = alpha_inv
    T_list = torch.eye(3)[None, None, None, None, :, :].type_as(img_list)
    T_list = T_list.repeat(1, vol_size, 2, 1, 1, 1)
    T_list[0, :, :, 0, 0, 0] = alpha_for
    T_list[0, :, :, 0, 1, 1] = alpha_for

    T_inv_list = T_inv_list.repeat(batch_size, 1, 1, 1, 1, 1)
    T_list = T_list.repeat(batch_size, 1, 1, 1, 1, 1)
 
    # prepare targets
    T_list = T_list[:, :, 0:1, :, :, :]
    
    out = {'T_inv_list': T_inv_list,
           'T_list': T_list}

    return out


def compute_volume_generation_params(attributes, img_list, 
                                     alpha_min, alpha_max, alpha_delta,
                                     shift_min, shift_max, shift_delta):

    # compute eta and theta list 
    if 'ttc' in attributes:
        alpha_list = torch.arange(alpha_min, alpha_max + alpha_delta - 1e-3, alpha_delta)
        alpha_list = alpha_list.type(torch.float32)[:, None].type_as(img_list)
        eta_delta = alpha_delta 
        eta_min_GT = alpha_list[0] - 0.5 * alpha_delta
        eta_max_GT = alpha_list[-1] + 0.5 * alpha_delta
    else:
        eta_delta = 0
        eta_min_GT = 0
        eta_max_GT = 0

    
    if 'of' in attributes:
        shift_list = torch.arange(shift_min, shift_max + shift_delta - 1e-3, shift_delta)
        shift_list = shift_list.type(torch.float32)[:, None].type_as(img_list)
        shiftx_list = shift_list.clone()
        shifty_list = shift_list.clone()
        
        of_delta = shift_delta 
        of_min_GT_x = shiftx_list[0] - 0.5 * shift_delta
        of_min_GT_y = shifty_list[0] - 0.5 * shift_delta
    else:
        of_delta = 0
        of_min_GT_x = 0
        of_max_GT_x = 0
        of_min_GT_y = 0
        of_max_GT_y = 0
    
    # transformations
    T_list = []
    T_inv_list = []
    T_list_in = []
    T_inv_list_in = []
    
    vol_ttc_size = 0
    vol_of_size = 0
    seg_ids = []
    if 'ttc' in attributes:
        ttc_params = compute_params_ttc_vol(img_list, alpha_list)
        T_ttc_list = ttc_params['T_list']
        T_ttc_inv_list = ttc_params['T_inv_list']
        
        T_list.append(T_ttc_list)
        T_inv_list.append(T_ttc_inv_list)
        T_list_in.append(T_ttc_list[:, ::3, ...].clone())
        T_inv_list_in.append(T_ttc_inv_list[:, ::3, ...].clone())
        vol_ttc_size = T_ttc_list.shape[1]
        seg_ids.append(torch.Tensor([[0]]).type(torch.cuda.LongTensor))
    

    
    if 'of' in attributes:
        of_params = compute_params_of_vol(img_list, shiftx_list, shifty_list)
        T_ofx_list = of_params['T_ofx_list']
        T_ofx_inv_list = of_params['T_ofx_inv_list']
        T_ofy_list = of_params['T_ofy_list']
        T_ofy_inv_list = of_params['T_ofy_inv_list']
        T_list.append(T_ofx_list)
        T_list.append(T_ofy_list)
        T_inv_list.append(T_ofx_inv_list)
        T_inv_list.append(T_ofy_inv_list)
        
        T_ofx_list_in = T_ofx_list[:, ::3, ...].clone()
        T_ofx_list_in[..., 0, 2]=T_ofx_list_in[..., 0, 2] / 3
        T_list_in.append(T_ofx_list_in)
        T_ofy_list_in = T_ofy_list[:, ::3, ...].clone()
        T_ofy_list_in[..., 1, 2]=T_ofy_list_in[..., 1, 2] / 3
        T_list_in.append(T_ofy_list_in)

        T_ofx_inv_list_in = T_ofx_inv_list[:, ::3, ...].clone()
        T_ofx_inv_list_in[..., 0, 2]=T_ofx_inv_list_in[..., 0, 2] / 3
        T_inv_list_in.append(T_ofx_inv_list_in)
        T_ofy_inv_list_in = T_ofy_inv_list[:, ::3, ...].clone()
        T_ofy_inv_list_in[..., 1, 2]=T_ofy_inv_list_in[..., 1, 2] / 3
        T_inv_list_in.append(T_ofy_inv_list_in)
        
        seg_ids.append(torch.Tensor([[1]]).type(torch.cuda.LongTensor))
        seg_ids.append(torch.Tensor([[2]]).type(torch.cuda.LongTensor))

        vol_of_size = T_ofx_list.shape[1]
    
    T_list = torch.cat(T_list, dim=1)
    T_inv_list = torch.cat(T_inv_list, dim=1)
    seg_ids = torch.cat(seg_ids, dim=1)

    out = {'T_list': T_list,
           'T_inv_list':T_inv_list,
           'T_list_in': T_list_in,
           'T_inv_list_in':T_inv_list_in,
           'eta_size':vol_ttc_size,
           'of_size':vol_of_size,
           'eta_min_GT': eta_min_GT,
           'of_min_GT':[of_min_GT_x, of_min_GT_y],
           'of_delta':of_delta,
           'eta_delta':eta_delta,
           'seg_ids':seg_ids}

    return out
