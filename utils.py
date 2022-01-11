#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:15:52 2021

@author: walidajalil


This file contains both the pre_process that we use as well as the one adopted from Chris Chute's Real NVP git repo for comparison:
    
    As well as his Squeeze operations function:
        
                   https://github.com/chrischute/real-nvp
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
from tqdm import tqdm, trange
import sys




def _pre_process(x):
    data_constraint = torch.tensor([0.95], dtype=torch.float32).cuda()
    
    noise = torch.rand_like(x).cuda()
    y = (x * 255. + noise) / 256.
    y = (2 * y - 1) * data_constraint
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
        - F.softplus((1. - data_constraint).log() - data_constraint.log())
    sldj = ldj.view(ldj.size(0), -1).sum(-1)

    return y, sldj

def pre_process(x):
    if not type(x) == torch.Tensor:
        
       raise ValueError("Data has to be of Type: PyTorch Tensor.")
       
    else:
        alpha = 0.1
        noise = torch.rand(x.size())
        y = (255*x + noise)
        
        #torch logit has an epsilon that can clamp the function.
        p = (alpha + (1-alpha)*(x/256))
        processed = torch.log(p/(1-p))
        
        #check that there isn't a problem here, since we add different numbers than chris chute
        det_jacobian = F.softplus(y) + F.softplus(-y) - F.softplus(torch.tensor([1. - 0.05]).log() - torch.tensor([0.05]).log())
        det_jacobian = det_jacobian.view(det_jacobian.size(0), -1).sum(-1)
        
    return processed, det_jacobian


class WNConv2d(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x




def loss(z, jacobian):

    prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
    
    prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) - np.log(256) * np.prod(z.size()[1:])
        
    nll = -(prior_ll + jacobian).mean()
    
    
    return nll


def squeeze_operation(x, unsqueeze = False, alternate = False):

    B, C, H, W = x.size()
    if alternate:
        if unsqueeze:
            C //= 4
        else:
            C = C

        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * C, C, 2, 2), dtype=x.dtype, device=x.device)
        
        for c_idx in range(C):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
            
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(C)]
                                        + [c_idx * 4 + 1 for c_idx in range(C)]
                                        + [c_idx * 4 + 2 for c_idx in range(C)]
                                        + [c_idx * 4 + 3 for c_idx in range(C)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if unsqueeze:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
        
        
    else:
         
        if not unsqueeze:    
            x = x.permute(0, 2, 3, 1)
            
            x = x.view(B, H//2, 2, W//2, 2, C)
            
            x = x.permute(0, 1, 3, 5, 2, 4)
            
            x = x.contiguous().view(B, H//2, W//2, 4*C)
            
            x = x.permute(0, 3, 1, 2)
            
            
        else:
            
            x = x.permute(0, 2, 3, 1)
    
            x = x.view(B, H, W, C//4, 2, 2)
    
            x = x.permute(0, 1, 4, 2, 5, 3)
    
            x = x.contiguous().view(B, H*2, W*2, C//4)
    
            x = x.permute(0, 3, 1, 2)
 
        
    return x
    



def checkerboard_mask(height, width, alternate = False, dtype=torch.float32,
                      device=None, requires_grad=False):

    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if alternate:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask


def channel_mask(B, C, H, W, alternate = False):
    
        mask = torch.ones([B, C // 2, H, W])
        non_mask = torch.zeros([B, C // 2, H, W])
        
        if not alternate:
            m = torch.cat((mask, non_mask), dim = 1)
        else:
            m = torch.cat((non_mask, mask), dim = 1)
            
        return m
        
        
        