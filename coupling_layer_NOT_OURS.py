#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This coupling layer is taken from Chris Chute's Real NVP git repo:
    https://github.com/chrischute/real-nvp



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
            in_channels //= 2
        
        #  in_channels, mid_channels, out_channels,
                     #num_blocks, kernel_size, padding, double_after_norm
        self.resnet_layer = ResNet(in_channels, mid_channels, num_blocks = num_resnet_blocks, out_channels = 2*in_channels, kernel_size = 3, padding = 1, double_after_norm = (self.mask_type == "Checkerboard"))
        self.scale_layer_normed = torch.nn.utils.weight_norm(Rescale(in_channels))
        
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))        
        
    def forward(self, x, jacobian = None, inverse = False):

        if self.mask_type == "Checkerboard":
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.alternate_mask, device=x.device)
            x_b = x * b

            st = self.resnet_layer(x_b)

            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if inverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                jacobian += s.reshape(s.size(0), -1).sum(-1)
        elif self.mask_type == "Channelwise":
            # Channel-wise mask
            if self.alternate_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)
            st = self.resnet_layer(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            # Scale and translate
            if inverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                jacobian += s.reshape(s.size(0), -1).sum(-1)

            if self.alternate_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, jacobian

                    
            
                
                    
                   
                
                
class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
    