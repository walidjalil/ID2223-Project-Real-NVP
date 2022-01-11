#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 6 19:35:44 2022

@author: walidajalil
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet
from utils import checkerboard_mask, channel_mask
import sys

class Affine_coupling(torch.nn.Module):  
    def __init__(self, num_resnet_blocks, in_channels, mid_channels, alternate_mask, mask_type):
        super(Affine_coupling, self).__init__()
        
        self.mask_type = mask_type
        self.alternate_mask = alternate_mask
        if self.mask_type == "Channelwise":
            in_channels // 2
        
        #  in_channels, mid_channels, out_channels,
                     #num_blocks, kernel_size, padding, double_after_norm
        self.resnet_layer = ResNet(in_channels, mid_channels, num_blocks = num_resnet_blocks, out_channels = 2*in_channels, kernel_size = 3, padding = 1, double_after_norm = (self.mask_type == "Checkerboard"))
        self.scale_layer_normed = torch.nn.utils.weight_norm(Scale_layer(in_channels))
        
        self.rescale = nn.utils.weight_norm(Scale_layer(in_channels))        
        
    def forward(self, x, jacobian = None, inverse = False):

        if self.mask_type == "Checkerboard":
                m = checkerboard_mask(x.size(2), x.size(3), self.alternate_mask).cuda()
                #print("x: ", x[0][0][0][0])
                x_masked = x * m
                s_t = self.resnet_layer(x_masked)
                split_size = s_t.size(1) // 2
                s, t = s_t.split(split_size, dim=1)
                s_scaled = self.scale_layer_normed(torch.tanh(s))
                #sys.exit("Exited")
                
                if not inverse:
                    y = x_masked + (1-m)*(x*torch.exp(s_scaled) + t)
                    jacobian = jacobian + s_scaled.reshape(s_scaled.size(0), -1).sum(-1)
                else:
                    y = x_masked + (1-m)*((x - t) * torch.exp(-1*s_scaled))
                    
        elif self.mask_type == "Channelwise":
            #channelwise masking
            B, C, H, W = x.size()
            m = channel_mask(B, C, H, W, self.alternate_mask == "alternate").cuda()
            x_masked = x * m
            s_t = self.resnet_layer(x_masked)
            split_size = s_t.size(1) // 2
            s, t = s_t.split(split_size, dim=1)
            s_scaled = self.scale_layer_normed(torch.tanh(s))
            
            if not inverse:
                y = x_masked + (1-m)*(x*torch.exp(s_scaled) + t)
                jacobian = jacobian + s_scaled.reshape(s_scaled.size(0), -1).sum(-1)
            else:
                y = x_masked + (1-m)*((x - t) * torch.exp(-1*s_scaled))
        
        return y, jacobian 
 
    
class Scale_layer(torch.nn.Module):
    def __init__(self, num_channels):
        super(Scale_layer, self).__init__()
        
        self.weight = torch.nn.Parameter(torch.ones(num_channels, 1, 1))
        
    
    def forward(self, x):
        scaled = self.weight * x
        
        return scaled
    
    
 
    
 
