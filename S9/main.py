# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:56:16 2021

@author: saina
"""

import torch



# from torchvision.models import vgg19

# vgg = vgg19(pretrained=True)        
# features_conv = vgg.features[:36]
# vgg.features

############ DATA & TRANSFORMS

from data import get_data
from device import get_device


device = get_device(force_cpu=False)
train_loader, test_loader = get_data(device,batch_size=64,data = 'cifar10')




##################### MODEL

from model import NetCifar2
from torchsummary import summary
model = NetCifar2().to(device)

summary(model, input_size=(3, 32, 32))




##################### RUN MODEL

from run import run_model

epochs = 20
regularization = {'l1_factor':0,'l2_factor':0}

model,train_trackers,test_trackers,incorrect_samples = run_model(model, train_loader, test_loader, epochs, device, **regularization)



# test_trackers['test_losses']


####################

