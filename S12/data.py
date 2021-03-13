# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:47:17 2021

@author: saina
"""


import torch
from torchvision import datasets

import albumentations as A
from albumentations.pytorch import ToTensor

import numpy as np


class Args:

    # Data Loading
    # ============

    train_batch_size = 64
    val_batch_size = 64
    num_workers = 4

    # Augmentation
    # ============
    horizontal_flip_prob = 0.2
    vertical_flip_prob = 0.0
    gaussian_blur_prob = 0.0
    rotate_degree = 20
    cutout = 0.3

    # Training
    # ========
    random_seed = 1
    epochs = 50
    learning_rate = 0.01
    momentum = 0.9
    lr_step_size = 25
    lr_gamma = 0.1

    # Evaluation
    # ==========
    sample_count = 25



import torch
from torchvision import datasets, transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels,transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]

        # Load data and get label
        X = self.list_IDs[index]
        y = self.labels[index]
        X = Image.fromarray(X)
        y = torch.Tensor(y)
        if self.transform:
          X = self.transform(X)

        return X, y


class Transforms:
    

    def __init__(self, train = True,  **transform_args):
        
        
        ## ARGS
        
        horizontal_flip_prob = transform_args['horizontal_flip_prob']
        vertical_flip_prob = transform_args['vertical_flip_prob']
        gaussian_blur_prob = transform_args['gaussian_blur_prob']
        rotate_degree = transform_args['rotate_degree']
        cutout = transform_args['cutout']
        cutout_height = transform_args['cutout_height']
        cutout_width = transform_args['cutout_width'] 
    
        
        mean=(0.5, 0.5, 0.5)
        std=(0.5, 0.5, 0.5)
        
        # Train phase transformations
         
        transforms_list = []
    
        if train:
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)]
            if cutout > 0:  # CutOut
                transforms_list += [A.CoarseDropout(
                    p=cutout, max_holes=1, fill_value=tuple([x * 255.0 for x in mean]),
                    max_height=cutout_height, max_width=cutout_width, min_height=1, min_width=1
                )]
          
    
        transforms_list += [
            # normalize the data with mean and standard deviation to keep values in range [-1, 1]
            # since there are 3 channels for each image,
            # we have to specify mean and std for each channel
            A.Normalize(mean=mean, std=std, always_apply=True),
            
            # convert the data to torch.FloatTensor
            # with values within the range [0.0 ,1.0]
            ToTensor()
        ]
    
    
        self.transform =  A.Compose(transforms_list)    

    def __call__(self, image):
        """Process and image through the data transformation pipeline.

        Args:
            image: Image to process.
        
        Returns:
            Transformed image.
        """

        image = np.array(image)
        image = self.transform(image=image)['image']
        return image



import cv2
import time
from skimage import io, transform
import numpy as np

path = './data/tiny_image_data/tiny-imagenet-200/'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [cv2.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i))) for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(cv2.imread( path + 'val/images/{}'.format(img_name)))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


# def get_data(device, transform_args ,batch_size = 32 , num_workers = 4 ):
    
#     if device.type == 'cpu':
#         cuda = False
#     else:
#         cuda = True
    
#     # dataloader arguments
#     dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)
    
#     train_transforms = Transforms(train=True, **transform_args)
#     test_transforms = Transforms(train = False , **transform_args)  

#     trainset = datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=train_transforms)
    
#     testset = datasets.CIFAR10(root='./data', train=False,
#                                            download=True, transform=test_transforms)

#     train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    
#     test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)


#     return train_loader,test_loader