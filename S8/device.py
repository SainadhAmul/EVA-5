# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:22:07 2021

@author: saina
"""

import torch


def get_device(force_cpu = True):
    
    if force_cpu:
        device = torch.device("cpu")
    
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

    return device
