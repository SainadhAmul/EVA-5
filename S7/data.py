# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:47:17 2021

@author: saina
"""


import torch
from torchvision import datasets, transforms



def set_transforms(augmentation = False , rotation = 5.0):
        
    
    
    ## BASIC TRANSFORMATIONS
    
    # Train phase transformations
    train_transforms = transforms.Compose([
    
        # convert the data to torch.FloatTensor with values within the range [0.0 ,1.0]
        transforms.ToTensor(),
    
        # normalize the data with mean and standard deviation
        # these values were obtained from the data statistics above
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test phase transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    
    if augmentation:
        train_transforms = [
            # Rotate image by 6 degrees
            transforms.RandomRotation((-rotation, rotation), fill=(1,))
        ] + train_transforms
    
    return train_transforms,test_transforms




def get_data(device , data = 'mnist' ,batch_size = 32 , num_workers = 4):
    
    if device.type == 'cpu':
        cuda = False
    else:
        cuda = True
    
    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)
        
    if data.lower() == 'mnist':
        
        train_transforms,test_transforms = set_transforms()
        
        train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
        test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
         
        
        
        # train dataloader
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        
        # test dataloader
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        

    elif data.lower() == 'cifar10':
        
        train_transforms,test_transforms = set_transforms()

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transforms)
        
        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=test_transforms)

        train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
        
        test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)


    return train_loader,test_loader