#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:09:12 2022

@author: walidajalil
"""
import os
import numpy as np
import torch
import torch.nn as nn
from NET import RealNVP
import torchvision

if torch.cuda.is_available():
    device = "cuda"



print("Initializing model: ")   
net = RealNVP()

checkpoint = torch.load("/home/walidajalil/ID_second/saved_models/model94.pt") 
net.load_state_dict(checkpoint['model_state_dict'])
net.to(device)

uniform_sample = torch.rand([16,3,32,32]).to(device)
normal_sample = torch.randn([1,3,32,32]).to(device)

y_uniform, _ = net(uniform_sample, inverse = True)
#y_normal, _ = net(normal_sample, inverse = True)
y = 12*((y_uniform - y_uniform.min()) / (y_uniform.max() - y_uniform.min())) - 6
y = torch.sigmoid(y)
#torchvision.utils.save_image(y,"image.png")

images_concat = torchvision.utils.make_grid(y, nrow=int(16 ** 0.5), padding=2, pad_value=255)
torchvision.utils.save_image(images_concat, 'samples.png')
