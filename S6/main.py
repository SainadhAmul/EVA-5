# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:56:16 2021

@author: saina
"""

import torch


############ DATA & TRANSFORMS

from data import get_data
from device import get_device


device = get_device(force_cpu=False)
train_loader, test_loader = get_data(device,batch_size=64)


##################### MODEL

from model import Net
from torchsummary import summary
model = Net().to(device)
summary(model, input_size=(1, 28, 28))



##################### MODEL WITH GBN

# from torchsummary import summary


# from model import Net2
# model = Net2().to(device)
# summary(model, input_size=(1, 28, 28))

##################### RUN MODEL

from run import run_model

epochs =5
regularization = {'l1_factor':0.0001,'l2_factor':0}

model,train_trackers,test_trackers,incorrect_samples = run_model(model, train_loader, test_loader, epochs, device, **regularization)



test_trackers['test_losses']


####################


# Initialize a figure
fig = plt.figure(figsize=(13, 11))

# Plot values
plain_plt, = plt.plot(plain)
l1_plt, = plt.plot(l1)
l2_plt, = plt.plot(l2)
l1_l2_plt, = plt.plot(l1_l2)

# Set plot title
plt.title(f'Validation {metric}')

# Label axes
plt.xlabel('Epoch')
plt.ylabel(metric)

# Set legend
location = 'upper' if metric == 'Loss' else 'lower'
plt.legend(
    (plain_plt, l1_plt, l2_plt, l1_l2_plt),
    ('Plain', 'L1', 'L2', 'L1 + L2'),
    loc=f'{location} right',
    shadow=True,
    prop={'size': 20}
)

# Save plot
fig.savefig(f'{metric.lower()}_change.png')