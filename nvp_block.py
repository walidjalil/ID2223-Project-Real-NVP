#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:29:53 2022

@author: walidajalil
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from affine_coupling import Affine_coupling
from utils import squeeze_operation

class NVP_Block(nn.Module):
    def __init__(self, number_scales, num_resnet_blocks, in_channels, mid_channels, recursion_idx):
        super(NVP_Block, self).__init__()
        ## Argumentws for affine_coupling: num_resnet_blocks, in_channels, mid_channels, unmask, mask_type
        self.recursion_idx = recursion_idx
        self.number_scales = number_scales

        self.first_couplings = torch.nn.ModuleList([Affine_coupling(num_resnet_blocks,
                                in_channels, mid_channels, alternate_mask = i%2  , mask_type = "Checkerboard") for i in range(3)])
        if recursion_idx == number_scales - 1:
            self.first_couplings.append(Affine_coupling(num_resnet_blocks,
                                    in_channels, mid_channels, alternate_mask = 1  , mask_type = "Checkerboard"))
        else: 
            self.second_couplings = torch.nn.ModuleList([Affine_coupling(num_resnet_blocks,
                                4*in_channels, 2*mid_channels, alternate_mask = i%2  , mask_type = "Channelwise") for i in range(3)])
            
            self.next_nvp_block = NVP_Block(number_scales, num_resnet_blocks, 2*in_channels, 2*mid_channels, recursion_idx + 1)
        
    def forward(self, x, jacobian, inverse):
        if not inverse:
           for layer in self.first_couplings:
               x, jacobian = layer(x, jacobian, inverse)
               
           if not self.recursion_idx == self.number_scales - 1:
               
               x = squeeze_operation(x, unsqueeze = False, alternate = False)
               
               for layer in self.second_couplings:
                   
                   x, jacobian = layer(x, jacobian, inverse)
                   
               x = squeeze_operation(x, unsqueeze = True, alternate = False)
               x = squeeze_operation(x, unsqueeze = False, alternate = True)
               size_split = x.size(1) // 2
               x1, x2 = x.split(size_split, dim = 1)
               x, jacobian = self.next_nvp_block(x1, jacobian, inverse)
               x = torch.cat((x, x2), dim = 1)
               x = squeeze_operation(x, unsqueeze = True, alternate = True)
        else:
            if not self.recursion_idx == self.number_scales - 1:
                x = squeeze_operation(x, unsequeeze = False, alternate= True)
                size_split = x.size(1) // 2
                x1, x2 = x.split(size_split, dim = 1) 
                x, jacobian = self.next_nvp_block(x1, jacobian, inverse)
                x = torch.cat((x, x2), dim = 1)
                x = squeeze_operation(x, unsqueeze = True, alternate = True)
                x = squeeze_operation(x, unsqueeze = False, alternate = False)
    
                for layer in reversed(self.second_couplings):
                       
                    x, jacobian = layer(x, jacobian, inverse)
                x = squeeze_operation(x, unsqueeze = True, alternate = False)

            for layer in reversed(self.first_couplings):
                x, jacobian = layer(x, jacobian, inverse)
            
            
            
        return x, jacobian