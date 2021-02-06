

import torch
import torch.nn as nn
import torch.nn.functional as F



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
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),
                        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate)
        )
        
        # Input: 8x8x32 | Output: 8x8x64 | RF: 32x32   jump = 3

        
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
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x