# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:38:57 2021

@author: saina
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import Net


def train(model, train_loader, criterion, optimizer, device, l1_factor =0,  **trackers):
      
    model.train()
    correct_classified = 0
    for batch_number , (x_train,y_train) in enumerate(train_loader):
        
        batch_number+=1
        
        x_train,y_train = x_train.to(device), y_train.to(device)

        # print(x_train.shape)
        pred = model.forward(x_train)

        # print(pred.shape)
        # print(y_train.shape)
        loss = criterion(pred,y_train)
        
        ## L1 LOSS
        if l1_factor > 0:  # Apply L1 regularization
            l1_criteria = nn.L1Loss(size_average=False)
            regularizer_loss = 0
            for parameter in model.parameters():
                regularizer_loss += l1_criteria(parameter, torch.zeros_like(parameter))
            loss += l1_factor * regularizer_loss
        
        #pred.argmax(dim=1, keepdim=True)
        #PyTorch .eq() function to do this, which compares the values in two tensors and if they match, returns a 1. If they don’t match, it returns a 0:
        #correct += pred.eq(target.view_as(pred)).sum().item()
        predicted = torch.max(pred.data ,1)[1]
        correct_classified += (predicted == y_train).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # train_loader.dataset.data / train_loader.batch_size
        
        if batch_number%100 == 0:
            
            acc = round((correct_classified.item())/(batch_number*train_loader.batch_size),5)
            print(f'(TRAIN) batch_number: {batch_number:4} Loss : {loss:4.4} Acc : {acc:4.5}')
    
    acc = round((correct_classified.item())/len(train_loader.dataset.data),5)
    
    prev_acc = trackers['train_acc']
    trackers['train_acc'] = prev_acc.append(acc)
    
    prev_losses = trackers['train_losses']
    trackers['train_losses'] = prev_losses.append(loss.item())  






def test(model, test_loader, criterion, device, incorrect_samples, **trackers):
    
    test_losses = []
    
    model.eval()
    with torch.no_grad():
        
        correct_classified = 0
        for batch_number , (x_test,y_test) in enumerate(test_loader):
        
            x_test,y_test = x_test.to(device), y_test.to(device)
            pred = model.forward(x_test)
            loss = criterion(pred,y_test)
            test_losses.append(loss)
            
            correct_classified += (torch.max(pred,1)[1] == y_test).sum()
            
            ## INCORRECT PRED SAMPLES !
            output = pred.argmax(dim=1, keepdim=True)
            result = output.eq(y_test.view_as(output))
        
            if len(incorrect_samples) < 25:
                for i in range(test_loader.batch_size):
                    if not list(result)[i]:
                        incorrect_samples.append({
                            'prediction': list(output)[i],
                            'label': list(y_test.view_as(output))[i],
                            'image': list(x_test)[i]
                        }) 
    
        avg_loss = torch.mean(torch.tensor(test_losses))
        acc = round(correct_classified.item()/len(test_loader.dataset.data),5)
        
        prev_acc = trackers['test_acc']
        trackers['test_acc'] = prev_acc.append(acc)
        
        prev_losses = trackers['test_losses']
        trackers['test_losses'] = prev_losses.append(avg_loss.item())  
        
        print('(TEST) Correct_classified : ' , correct_classified.item() ,' of 10000')
        print(f'(TEST) Loss : {avg_loss:4.4} Acc : {acc:4.5}')
        print('\n','*'*60 , '\n')
        

     
        
     
        
        
def run_model(model, train_loader, test_loader, epochs, device, **regularization):
        
    # model = Net().to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    l2_factor = regularization['l2_factor']
    l1_factor = regularization['l1_factor']
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=l2_factor)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.15)
    
    ## TRACKERS
    train_losses = []
    train_acc = []
    train_trackers = {'train_acc':train_acc,'train_losses':train_losses}
    
    test_acc = []
    test_losses = []
    test_trackers = {'test_acc':test_acc,'test_losses':test_losses}
    
    incorrect_samples = []
    
    ## Model RUN!
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model,train_loader, criterion, optimizer, device,l1_factor =l1_factor, **train_trackers)
        scheduler.step()
        test(model, test_loader, criterion, device, incorrect_samples, **test_trackers)
        
    return model,train_trackers,test_trackers,incorrect_samples

        