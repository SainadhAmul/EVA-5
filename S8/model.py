# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:33:43 2021

@author: saina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



#change the architecture to C1C2C3C40 (basically 3 MPs)
#total RF must be more than 44

# rf = rf-in + (K-1) * jump

# jump-out = jump-in * stride

# n-out = ((n-in +2p -kernelrsize)/stride) +1

# one of the layers must use Depthwise Separable Convolution
# one of the layers must use Dilated Convolution

class NetCifar(nn.Module):
    
    
    def __init__(self):
        super(NetCifar, self).__init__()
        
        dropout_rate = 0.01
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3 , padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3 , padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )
        # Input: 32x32x3 | Output: 32x32x16 | RF: 5x5   jump = 1
        
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # Input: 32x32x32 | Output: 16x16x32 | RF: 6x6   jump = 1-->2  
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate)
        )
        
        # Input: 16x16x16 | Output: 16x16x32 | RF: 14x14   jump = 2

        self.pool2 = nn.MaxPool2d(2, 2)
        # Input: 16x16x32 | Output: 8x8x32 | RF: 16x16   jump = 2--> 4
        
        self.convblock3 = nn.Sequential(
            
            # grouped convilution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 32x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),  # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )
        
        # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32   jump = 3

        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            # Input: 8x8x64 | Output: 4x4x64 | RF: 36x36   jump = 4--> 8       
            
            # Pointwise convolution to reduce channels
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  
            # Input: 4x4x64 | Output: 4x4x32 | RF: 36x36
        )

        self.convblock4_1 = nn.Sequential(
            
            # Normal Conv
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate)
            )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 52x52
        
        
        self.convblock4_2 = nn.Sequential(
            
            # dilated convilution
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,dilation=2,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate)
            )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 68x68  (36 + 2* 16)
            
        self.convblock4 = nn.Sequential(
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 68x68 / 84x84
        
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 4x4x64 | Output: 1x1x64 | RF:  92x92 / 108x108 

        self.fc = nn.Sequential(
            nn.Linear(64, 10)
        )       
        
        

    def forward(self, x):

        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        
        x = self.convblock3(x)
        
        x = self.transblock1(x)
        # dilated conv
        
        x1 = self.convblock4_1(x[:,:16,:,:])
        x2 = self.convblock4_2(x[:,16:,:,:])
        
        x = torch.cat( (x1 ,x2) , 1)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x





############################################






class NetCifar2(nn.Module):
    
    
    def __init__(self):
        super(NetCifar2, self).__init__()
        
        dropout_rate = 0.01
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3 , padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3 , padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )
        # Input: 32x32x3 | Output: 32x32x16 | RF: 5x5   jump = 1
        
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # Input: 32x32x32 | Output: 16x16x32 | RF: 6x6   jump = 1-->2  
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate)
        )
        
        # Input: 16x16x16 | Output: 16x16x32 | RF: 14x14   jump = 2

        self.pool2 = nn.MaxPool2d(2, 2)
        # Input: 16x16x32 | Output: 8x8x32 | RF: 16x16   jump = 2--> 4
        


        self.convblock3_1 = nn.Sequential(
            
            # Normal Conv
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
            )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 52x52
        
        
        self.convblock3_2 = nn.Sequential(
            
            # dilated convilution
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3,dilation=2,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
            )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 68x68  (36 + 2* 16)
            
        self.convblock3 = nn.Sequential(
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_rate)
        )
        # Input: 4x4x16 | Output: 4x4x16 | RF: 68x68 / 84x84
        
        
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            # Input: 8x8x64 | Output: 4x4x64 | RF: 36x36   jump = 4--> 8       
            
            # Pointwise convolution to reduce channels
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)  
            # Input: 4x4x64 | Output: 4x4x32 | RF: 36x36
        )
        
        
        self.convblock4 = nn.Sequential(
            
            # grouped convilution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding=1),  # Input: 8x8x32 | Output: 8x8x32 | RF: 32x32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),  # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_rate)
        )
        
        # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32   jump = 3


        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 4x4x64 | Output: 1x1x64 | RF:  92x92 / 108x108 

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )       
        
        

    def forward(self, x):

        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        
        # dilated conv
        
        x1 = self.convblock3_1(x[:,:16,:,:])
        x2 = self.convblock3_2(x[:,16:,:,:])
        
        x = torch.cat( (x1 ,x2) , 1)
        x = self.convblock3(x)
        
        
        x = self.transblock1(x)
        
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x












#######################################   MNIST 99.3

class Net(nn.Module):
    def __init__(self):
        """ This function instantiates all the model layers """
        super(Net, self).__init__()

        dropout_rate = 0.01

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate)
        )  # Input: 28x28x1 | Output: 26x26x8 | RF: 3x3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate)
        )  # Input: 26x26x8 | Output: 24x24x8 | RF: 5x5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )  # Input: 24x24x8 | Output: 22x22x16 | RF: 7x7

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )  # Input: 22x22x16 | Output: 20x20x16 | RF: 9x9

        self.pool = nn.MaxPool2d(2, 2)  # Input: 20x20x16 | Output: 10x10x16 | RF: 10x10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )  # Input: 10x10x16 | Output: 8x8x16 | RF: 14x14

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )  # Input: 8x8x16 | Output: 6x6x16 | RF: 18x18

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate)
        )  # Input: 6x6x16 | Output: 6x6x10 | RF: 18x18

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 6x6x10 | Output: 1x1x10 | RF: 28x28
    
    def forward(self, x):
        """ This function defines the network structure """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
############################## WITH GBN


from gbn import GhostBatchNorm
from functools import partial

from gbn import GBN

class Net2(nn.Module):
    def __init__(self):
        """ This function instantiates all the model layers """
        super(Net2, self).__init__()

        dropout_rate = 0.01
        BatchNorm = partial(GhostBatchNorm, num_splits=4, weight=False)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            BatchNorm(8),
            nn.Dropout(dropout_rate)
        )  # Input: 28x28x1 | Output: 26x26x8 | RF: 3x3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            BatchNorm(8),
            nn.Dropout(dropout_rate)
        )  # Input: 26x26x8 | Output: 24x24x8 | RF: 5x5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            BatchNorm(16),
            nn.Dropout(dropout_rate)
        )  # Input: 24x24x8 | Output: 22x22x16 | RF: 7x7

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            BatchNorm(16),
            nn.Dropout(dropout_rate)
        )  # Input: 22x22x16 | Output: 20x20x16 | RF: 9x9

        self.pool = nn.MaxPool2d(2, 2)  # Input: 20x20x16 | Output: 10x10x16 | RF: 10x10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            BatchNorm(16),
            nn.Dropout(dropout_rate)
        )  # Input: 10x10x16 | Output: 8x8x16 | RF: 14x14

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            BatchNorm(16),
            nn.Dropout(dropout_rate)
        )  # Input: 8x8x16 | Output: 6x6x16 | RF: 18x18

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1),
            nn.ReLU(),
            BatchNorm(10),
            nn.Dropout(dropout_rate)
        )  # Input: 6x6x16 | Output: 6x6x10 | RF: 18x18

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 6x6x10 | Output: 1x1x10 | RF: 28x28
    
    def forward(self, x):
        """ This function defines the network structure """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)