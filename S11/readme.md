# One Cycle Learning Rate Policy for Keras
Implementation of One-Cycle Learning rate policy from the papers by Leslie N. Smith.

- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        
# What is One Cycle Learning Rate
It is the combination of gradually increasing learning rate, and optionally, gradually decreasing the momentum during the first half of the cycle, then gradually decreasing the learning rate and optionally increasing the momentum during the latter half of the cycle. 

Finally, in a certain percentage of the end of the cycle, the learning rate is sharply reduced every epoch. 

The Learning rate schedule is visualized as : 

<img src="https://github.com/titu1994/keras-one-cycle/blob/master/images/one_cycle_lr.png?raw=true" height=50% width=100%> 

The Optional Momentum schedule is visualized as : 

<img src="https://github.com/titu1994/keras-one-cycle/blob/master/images/one_cycle_momentum.png?raw=true" height=50% width=100%>

# Usage

The author recommends doing one cycle of learning rate of 2 steps of equal length. We choose the maximum learning rate
using a range test. We use a lower learning rate as 1/5th or 1/10th of the maximum learning rate. We go from a lower
learning rate to a higher learning rate in step 1 and back to a lower learning rate in step 2. We pick this cycle length slightly
lesser than the total number of epochs to be trained. And in the last remaining iterations, we annihilate the learning
rate way below the lower learning rate value(1/10 th or 1/100 th).


# Then why use One Cycle?

It reduces the time it takes to reach "near" to your accuracy. 
It allows us to know if we are going right early on. 
It let us know what kind of accuracies we can target with a given model.
It reduces the cost of training. 
It reduces the time to deploy!



## Finding a good learning rate
Minimum and Maximum Boundary Values
 

LR Range Test:  Run your model for several epochs while letting the learning rate increase linearly between low
and high LR values.

<img src="![image](https://user-images.githubusercontent.com/45446030/114859530-81d3cb00-9e08-11eb-897a-7fa7df760147.png)>

STEP-SIZE: CIFAR10 has 50,000 training images. For the batch size 100, this means each cycle would be
50000/100 = 500 iterations. 

However, results show, that it is often SLIGHTLY good to set stepsize  equal to 2-10 times the number of iterations.

 

**Note : When using this, be careful about setting the learning rate, momentum and weight decay schedule. The loss plots will be more erratic due to the sampling of the validation set.**





## Training with `OneCycleLR`
Once we find the maximum learning rate, we can then move onto using the `OneCycleLR` callback with SGD to train our model.

```python - Pytorch
    scheduler = lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=24, steps_per_epoch= 64, pct_start = 0.2)
    
    ## Model RUN!
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model,train_loader, criterion, optimizer, device,l1_factor =l1_factor, **train_trackers)
        scheduler.step()
        test(model, test_loader, criterion, device, incorrect_samples, **test_trackers)
```

There are many parameters, but a few of the important ones : 

learning_rate, epochs, steps_per_epoch, pct_start

- optimizer (Optimizer) – Wrapped optimizer.
- max_lr (float or list) – Upper learning rate boundaries in the cycle for each parameter group.
- total_steps (int) – The total number of steps in the cycle. Note that if a value is not provided here, then it must be inferred by providing a value for epochs and - steps_per_epoch. Default: None
- epochs (int) – The number of epochs to train for. This is used along with steps_per_epoch in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None
- steps_per_epoch (int) – The number of steps per epoch to train for. This is used along with epochs in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None
- pct_start (float) – The percentage of the cycle (in number of steps) spent increasing the learning rate. Default: 0.3
- anneal_strategy (str) – {‘cos’, ‘linear’} Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing. Default: ‘cos’
- cycle_momentum (bool) – If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’. Default: True
- base_momentum (float or list) – Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; at the peak of a cycle, momentum is ‘base_momentum’ and learning rate is ‘max_lr’. Default: 0.85
- max_momentum (float or list) – Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum). Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is ‘max_momentum’ and learning rate is ‘base_lr’ Default: 0.95


Note also that the total number of steps in the cycle can be determined in one of two ways (listed in order of precedence):

A value for total_steps is explicitly provided.
A number of epochs (epochs) and a number of steps per epoch (steps_per_epoch) are provided. In this case, the number of total steps is inferred by total_steps = epochs * steps_per_epoch
You must either provide a value for total_steps or provide a value for both epochs and steps_per_epoch.

# Results

Quicker jumps in accuracy are evident due to onecycle lr implementation

Accuracy: 
![acc](https://user-images.githubusercontent.com/45446030/114860820-1854bc00-9e0a-11eb-8e0f-1f19a12f350f.png)

Loss:
![loss](https://user-images.githubusercontent.com/45446030/114860832-1ab71600-9e0a-11eb-8494-1ffbb8d0e596.png)





