#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:54:52 2021

@author: walidajalil
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
from utils import pre_process, _pre_process
from nvp_block import NVP_Block

class RealNVP(torch.nn.Module):  
    def __init__(self, number_scales = 2, num_resnet_blocks = 8, in_channels = 3, mid_channels = 64):
        
        super(RealNVP, self).__init__()
        self.recursion_idx = 0
        self.first_block = NVP_Block(number_scales, num_resnet_blocks, in_channels, mid_channels, self.recursion_idx)

    def forward(self, x, inverse = False):
        
        x, jacobian = _pre_process(x)
 
        x, jacobian = self.first_block(x, jacobian, inverse)
        
        return x, jacobian