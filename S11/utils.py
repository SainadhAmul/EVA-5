# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:09:22 2021

@author: saina
"""

# !pip install torch-lr-finder


from torch_lr_finder import LRFinder

def plot_lr_(model, train_loader, test_loader, optimizer, criterion ,device = 'cpu' , step_mode = "linear" ):
    
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    # lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=1, num_iter=100, step_mode = step_mode)
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
    
    
