# Copyright 2021 NVIDIA CORPORATION & AFFILIATES
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import argparse 
import time 
import torch.backends.cudnn as cudnn

from models.PSMNet import conv2d
from models.PSMNet import conv2d_lrelu

from models.RegRefine2D import RegRefineNet

__all__ = ['regrefinenet', 'segrefinenet']


"""
Refinement network for regression output.
Takes concatenated input image and the estimated map to generate refined map.
Generates refined output using input image as guide.
"""
def regrefinenet(options, data=None):
    
    print('==> USING RegRefineNet')
    for key in options:
        if 'regrefinenet' in key:
            print('{} : {}'.format(key, options[key]))
    
    model = RegRefineNet(out_planes = options['regrefinenet_out_planes'])

    if data is not None:
        model.load_state_dict(data['state_dict'])
    
    return model


"""
Binary segmentation refinement network.
Takes as input high resolution features of input image and the coarse segmentation map.
Generates refined output using input image as guide.
"""
class SegRefineNet(nn.Module):
    def __init__(self, in_planes = 17, out_planes = 8, num_layers=1):
        super(SegRefineNet, self).__init__()

        if num_layers==2:
            self.conv1 = nn.Sequential(
                    conv2d_lrelu(in_planes, out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(in_planes, out_planes, kernel_size=3, stride=1, pad=1))
        elif num_layers==3:
            self.conv1 = nn.Sequential(
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1))
        elif num_layers==4:
            self.conv1 = nn.Sequential(
                    conv2d_lrelu(in_planes , out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1),
                    conv2d_lrelu(out_planes, out_planes, kernel_size=3, stride=1, pad=1))
        else:
            self.conv1 = nn.Sequential(
                    conv2d_lrelu(in_planes, out_planes, kernel_size=3, stride=1, pad=1))

        self.classif1 = nn.Conv2d(out_planes, 1, kernel_size=3, padding=1, stride=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, input):

        output0 = self.conv1(input) 
        output  = self.classif1(output0)
        
        return output


def segrefinenet(options, data=None):
    
    print('==> USING SegRefineNet')
    for key in options:
        if 'segrefinenet' in key:
            print('{} : {}'.format(key, options[key]))
    
    model = SegRefineNet(in_planes = options['segrefinenet_in_planes'],
                         out_planes = options['segrefinenet_out_planes'],
                         num_layers = options['segrefinenet_num_layers'])

    if data is not None:
        model.load_state_dict(data['state_dict'])
    
    return model
