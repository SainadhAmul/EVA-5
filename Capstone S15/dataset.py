

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:36:54 2021

@author: rampfire
"""
# import warnings
# warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
# from . import data_transformations as data_transforms
import logging
import albumentations as A
from albumentations.pytorch import ToTensor



transforms_list = [
    # normalize the data with mean and standard deviation to keep values in range [-1, 1]
    # since there are 3 channels for each image,
    # we have to specify mean and std for each channel

    # convert the data to torch.FloatTensor
    # with values within the range [0.0 ,1.0]
    # A.Resize(416,416,1),
    tempToTensor()
]

transform_composed =  A.Compose(transforms_list)



class TSAIDataset(Dataset):
    def __init__(self, list_path, img_size, is_training,data_dir_path = None, is_debug=False):
        self.img_files = []
        self.label_files = []
        self.depth_files = []
        for path in open(list_path, 'r'):
#            if data_dir_path is not None:
#                pass
#            else:
            path = path.replace("./data/customdata/images/",data_dir_path).strip()
            # print(path)
            label_path = path.replace('images', 'labels').replace('.png', '.txt').replace(
                '.jpg', '.txt').strip()
            depth_path = path.replace('images','depth_images').replace('.jpg','.png').strip()
            # print(depth_path)
            if os.path.isfile(label_path):
#                print(path)
                self.img_files.append(path.strip())
                self.label_files.append(label_path.strip())
                self.depth_files.append(depth_path.strip())
            else:
#                logging.info("no label found. skip it: {}".format(path))
                pass
#        logging.info("Total images: {}".format(len(self.img_files)))
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.is_debug = is_debug

        #  transforms and augmentation
#        self.transforms = data_transforms.Compose()
        # if is_training:
        #     self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
#        self.transforms.add(data_transforms.ResizeImage(self.img_size))
#        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        if os.path.exists(self.depth_files[index]):
            depth_data = cv2.imread(self.depth_files[index],0)
        else:
            # logging.info("label does not exist: {}".format(label_path))
            depth_data = np.zeros((416,416), np.float32)

        img = cv2.resize(img, (416,416),interpolation = cv2.INTER_AREA)

        depth_data = cv2.resize(depth_data, (416,416),interpolation = cv2.INTER_AREA)
        # print(labels.shape)
        sample = {'image': img, 'label': labels}
        sample = totensor_object(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        sample["depth"] = depth_data
        return sample

    def __len__(self):
        return len(self.img_files)





class tempToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}
