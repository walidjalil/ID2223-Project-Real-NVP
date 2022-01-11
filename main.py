#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:21:23 2021

@author: walidajalil
"""
import os
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
from tqdm import tqdm, trange
import sys
from NET import RealNVP
from utils import loss
from torch.autograd import Variable
import norm_util
import optim_util

PATH_save_models = " Put the path to the folder where you will save your models HERE ----"
PATH_load_models = " put the path here if you want to load a model AND model name which ends with .pt"


use_amp = True

if not os.path.isdir(PATH_save_models):
    os.makedirs(PATH_save_models)

if torch.cuda.is_available():
    device = "cuda"

training_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
    
trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4)


test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()
    ]))
    
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

print("Initializing model: ")   
net = RealNVP()
checkpoint = torch.load("/home/walidajalil/ID_second/saved_models/model19.pt") 
net.load_state_dict(checkpoint['model_state_dict'])
net.cuda()

#get the parameter names from the model so that we can seperate out the ones that need L2 regularization and the ones that don't.
# Will be saved as a dictionary with keys. 
norm_params = []
standard_params = []
for name, parameter in net.named_parameters():
    if name.endswith("weight_g"):
        norm_params.append(parameter)
    else:
        standard_params.append(parameter)

param_dictionaries = [{'name': 'normalized', 'params': norm_params, 'weight_decay': 5e-5},
                {'name': 'standard', 'params': standard_params}]


optimizer = torch.optim.Adam(param_dictionaries)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler.load_state_dict(checkpoint["scaler"])
epoch_loss_list = []
validation_loss_epoch = []

for epoch in range(0,100):
    print("Currently at Epoch: ", epoch)
    net.train()
    loss_list = []
    j = 0
    start = time.time()
    for batch, label in trainloader:
        j = j+1
        batch = batch.cuda()
        
        with torch.cuda.amp.autocast(enabled=use_amp):    
            y, jacobian = net(batch, inverse = False)
            loss_output = loss(y, jacobian)
        scaler.scale(loss_output).backward()
        scaler.unscale_(optimizer)
        loss_list.append(loss_output.item())
        #loss_output.backward()
        optim_util.clip_grad_norm(optimizer, max_norm = 100)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if (j+1)%100 == 0:
            
            print("Train Loss: ", loss_output.item())
            print(" ")
            print("Bits per dim: ", loss_output.item() / (np.log(2)*3072))
        if epoch == 0 and j == 1:
            path = PATH_save_models + str(epoch) + ".pt"
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler': scaler.state_dict()}, path)
    print(" ")
    print("Time to go through 1 epoch: ")
    
    print(time.time() - start)
    print(" ")
    if (epoch+1) % 5 == 0:
        path = PATH_save_models + str(epoch) + ".pt"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': scaler.state_dict()}, path)
    epoch_loss_list.append(sum(loss_list)/len(loss_list))
    print(" ")
    print("--------------------------")
    print("Average Training loss this epoch: ")
    print(sum(loss_list)/len(loss_list))
    print(" ")
    print("Bits per dimension this epoch: ")
    print( (sum(loss_list)/len(loss_list)) / (np.log(2)*3072))
    
    if (epoch)%5 == 0:
        validation_loss = []
        net.eval()
        with torch.no_grad():
            for batch_test, label_test in testloader:
                batch_test = batch_test.cuda()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    y_test, jacobian_test = net(batch, inverse = False)
                    loss_output_test = loss(y_test, jacobian_test)
                validation_loss.append(loss_output_test.item())
        val_loss = sum(validation_loss)/(len(validation_loss))
        print("------------------------------------------------")
        print("Epoch: ", epoch)
        print(" ")
        print("Validation Loss Average: ", loss_output_test.item())
        print(" ")
        print("Validation - Bits per dim: ", val_loss / (np.log(2)*3072))
        print(" ")
        print("-------------------------------------------------")
            
