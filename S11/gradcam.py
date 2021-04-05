# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:27:25 2021

@author: saina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

import matplotlib.pyplot as plt


class ModelGradCam(nn.Module):
    
    def __init__(self,model):
        super().__init__()
        
        # get the pretrained resnet network
        self.res = model
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(*list(self.res.children())[:-2])
        
        # # get the max pool of the features stem
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the resnet
        self.classifier1 = list(self.res.children())[-2:][0]
        self.classifier2 = list(self.res.children())[-2:][1]

        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.classifier1(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    


def gradcam_out(model,img_d,device):
    
    ref = {0: 'airplane',1: 'automobile',2: 'bird',3: 'cat',
             4: 'deer',5: 'dog',6: 'frog',7: 'horse',
             8: 'ship',9: 'truck'}
    
    img = img_d['image']
    label = img_d['label'].item()
    predicted = img_d['prediction'].item()
    
    # img = img.to(device)
    
    # initialize the resnet model
    res = ModelGradCam(model)
    
    # set the evaluation mode
    res.eval()
    
    x = img.view(1,3,32,32)
    
    # get the most likely prediction of the model
    pred = res(x)
    
    # get the gradient of the output with respect to the parameters of the model
    pred[:,label].backward()
    
    # pull the gradients out of the model
    gradients = res.get_activations_gradient()
    
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # get the activations of the last convolutional layer
    activations = res.get_activations(x).detach()
    
    # weight the channels by corresponding gradients
    
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]
        
        
    # weight the channels by corresponding gradients
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    # heatmap.shape
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)
    # heatmap.shape
    
    heatmap /= torch.max(heatmap)
    
    img = img_d['image']
    img_arr = np.transpose(img.cpu().data.numpy() , (1,2,0))
    img_arr = ((img_arr - img_arr.min()) * (1/(img_arr.max() - img_arr.min()) * 255)).astype('uint8')
    
    heatmap_numpy_resized = cv2.resize(heatmap.cpu().data.numpy(), (img_arr.shape[0], img_arr.shape[1]))
    heatmap_rescaled = np.uint8(255 * heatmap_numpy_resized)

    # plt.imshow(heatmap_rescaled)
    heatmap_final = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    superimposed_img = heatmap_final * 0.4 + img_arr
    cv2.imwrite('C:\\Users\\saina\\Documents\\EVA\\S10\\Incorrect_GC\\' +ref[label] + '_' + ref[predicted]+ '.jpg', superimposed_img)
    print('Image Written')
    
    return superimposed_img, ref[label], ref[predicted]