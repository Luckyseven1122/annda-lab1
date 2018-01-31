#!/usr/bin/env python3

import itertools
import collections
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from animator import Animator
from datageneration import mackey_glass

def plot_valid_dma():
    fig = plt.figure(figsize=(10, 6))
    ax_valid = fig.add_subplot(211)
    ax_dma = fig.add_subplot(212)
    ax_valid.plot(valid_over_epoch)
    ax_dma.plot(dma_over_epoch)
    plt.savefig('dma_valid.png')

def save_to_csv(filepath, varlist):
    try:
        with open(filepath, 'r+') as data_file:
            all_lines = list(data_file.read())

    except IOError:
        all_lines = []
    all_lines.append(''.join(str(varlist)))
    with open(filepath, 'w') as data_file:
        [data_file.write(s) for s in all_lines]

def plot_validation_set():
    fig = plt.figure(figsize=(10, 6), dpi=220)
    tax = fig.add_subplot(211)
    target = pp.revert(validation_target_noiseless)
    pred = pp.revert(final_validation)
    tax.plot(t_valid, target.data.tolist(), c='#000099')
    tax.plot(t_valid, pred.data.tolist(), c='#990000')
    eax = fig.add_subplot(212)
    pred = pred.view(target.size())
    eax.plot(t_valid, (pred - target).data.tolist(), c='#ff5050')
    plt.savefig('validation_set.png')



class EarlyStop:

    MOVING_AVERAGE_RANGE = 250

    INCREASING_THRESHOLD = 1e-6
    INCREASING_MAX_ITER = 500

    STOPPED_LEARNING_THRESHOLD = 5e-6
    STOPPED_LEARNING_MAX_ITER = 500

    NORMAL_EXIT_FLAG = 0
    STOPPED_LEARNING_EXIT_FLAG = -1
    INCREASING_EXIT_FLAG = -2
    BREAK_FLAGS = (
        STOPPED_LEARNING_EXIT_FLAG,
        INCREASING_EXIT_FLAG
    )

    EXIT_FLAG_STR = {
        STOPPED_LEARNING_EXIT_FLAG: 'Validation error stable',
        INCREASING_EXIT_FLAG: 'Increasing validation error'
    }

    def __init__(self):
        self.increasing_counter = 0
        self.stopped_learning_counter = 0
        self.global_min = float('inf')

        init_moving_average = [1]*self.MOVING_AVERAGE_RANGE
        self.prev_x = collections.deque(init_moving_average)
        self.ma_prev = 1
        self.dma = 0

    def diff_moving_average(self, new_val):
        self.prev_x.popleft()
        self.prev_x.append(new_val)
        new_ma = np.mean(self.prev_x)
        self.dma = new_ma - self.ma_prev
        self.ma_prev = new_ma

    def check_for_early_stop(self, new_val):

        set_new_global_min = self.check_global_min(new_val)
        self.diff_moving_average(new_val)
        increasing_raised = self.check_increasing()
        stopped_learning_raised = self.check_stopped_learning()

        if increasing_raised:
            exit_flag = self.INCREASING_EXIT_FLAG
        elif stopped_learning_raised:
            exit_flag = self.STOPPED_LEARNING_EXIT_FLAG
        else:
            exit_flag = self.NORMAL_EXIT_FLAG

        return set_new_global_min, exit_flag

    def check_global_min(self, new_val):

        if new_val <= self.global_min:
            self.global_min = new_val
            return True
        return False

    def check_increasing(self):

        if self.dma >= self.INCREASING_THRESHOLD:
            self.increasing_counter += 1
        else:
            self.increasing_counter = 0

        if self.increasing_counter >= self.INCREASING_MAX_ITER:
            return True
        return False

    def check_stopped_learning(self):

        if abs(self.dma) <= self.STOPPED_LEARNING_THRESHOLD:
            self.stopped_learning_counter += 1
        else:
            self.stopped_learning_counter = 0

        if self.stopped_learning_counter >= self.STOPPED_LEARNING_MAX_ITER:
            return True
        return False

class Preprocessor:

    def __init__(self):
        self.mean = 0
        self.std = 0

    def preprocess(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean)/self.std

    def revert(self, data):
        return data * self.std + self.mean


sigma = 0.03

pp = Preprocessor()

mglass = mackey_glass(1.5, 1500)
mglass = pp.preprocess(mglass)

mglass_noiseless = mglass.copy()
mglass += np.random.normal(0, sigma, mglass.shape)

t_train = np.arange(300, 800)
t_valid = np.arange(801, 1300)
t_test = np.arange(1301, 1500)
x_train = torch.t(Variable(torch.Tensor([
    mglass[t_train-20],
    mglass[t_train-15],
    mglass[t_train-10],
    mglass[t_train-5],
    mglass[t_train]
])))
x_valid = torch.t(Variable(torch.Tensor([
    mglass[t_valid-20],
    mglass[t_valid-15],
    mglass[t_valid-10],
    mglass[t_valid-5],
    mglass[t_valid]
])))

target_train = Variable(torch.Tensor(mglass[t_train+5]))
target_valid = Variable(torch.Tensor(mglass[t_valid+5]))
validation_target_noiseless = Variable(torch.Tensor(mglass_noiseless[t_valid+5]))

animator = Animator(t_train, pp.revert(mglass[t_train+5]),
                    t_valid, pp.revert(mglass[t_valid+5]))

learning_rate = 0.5
alpha = 0.9
alphainv = 1 - alpha
regularization = 0.1
hidden_size = 3
hidden_size_2 = 3

neural_net = nn.Sequential(
    nn.Linear(5, hidden_size),
    nn.Sigmoid(),
    nn.Linear(hidden_size, hidden_size_2),
    nn.Sigmoid(),
    nn.Linear(hidden_size_2, 1)
)

loss_over_epoch = []
valid_over_epoch = []
dma_over_epoch = []

optimal_params = []
for param in neural_net.parameters():
    param.register_hook(lambda grad: alphainv * grad)
    optimal_params.append(param)
    param.grad = Variable(torch.zeros(param.size()))

l2_norm = Variable(torch.FloatTensor(1), requires_grad=True)
w0 = 5
mse_loss = nn.MSELoss()

early_stopper = EarlyStop()

noise_mean = torch.zeros(x_train.size())
noise_std = 0.01 * torch.ones(x_train.size())

for epoch in itertools.count():

    # l2 norm over all parameters for regularization
    l2_norm.data = torch.Tensor([0])
    for param in neural_net.parameters():
        for val in param.view(-1 , 1):
            l2_norm.data += (val.data/w0)**2/(1 + (val.data/w0)**2)
        # l2_norm.data += param.norm(2).data

    # apply small white noise on training data
    x_train_noisy = x_train + Variable(torch.normal(noise_mean, noise_std))

    # errors over training data
    pred_train = neural_net.forward(x_train_noisy)
    loss = mse_loss(pred_train, target_train) + regularization * l2_norm
    loss_over_epoch.append(loss.data[0])

    # errors over validation data
    pred_valid = neural_net.forward(x_valid)
    valid_error = mse_loss(pred_valid, target_valid) + regularization * l2_norm
    valid_over_epoch.append(valid_error.data[0])

    for param in neural_net.parameters():
        param.grad *= alpha
    loss.backward()
    for param in neural_net.parameters():
        param.data -= learning_rate * param.grad.data

    if epoch % animator.EPOCHS_PER_FRAME == 0:
        print('Epoch {:4d}, loss: {:6.5f}, validation error: {:6.5f}'.format(
               epoch, loss.data[0], valid_error.data[0]))
        pred_train_list = pp.revert(pred_train).data.tolist()
        pred_valid_list = pp.revert(pred_valid).data.tolist()
        animator.add_epoch(epoch, pred_train_list, pred_valid_list)

    set_global_min, stop_flag = early_stopper.check_for_early_stop(valid_error.data[0])
    dma_over_epoch.append(early_stopper.dma)

    if set_global_min:
        for optimal, param in zip(optimal_params, neural_net.parameters()):
            optimal = param

    if stop_flag in EarlyStop.BREAK_FLAGS:

        for optimal, param in zip(optimal_params, neural_net.parameters()):
            param = optimal

        print('Epoch {:4d}, loss: {:6.5f}, validation error: {:6.5f}'.format(
               epoch, loss.data[0], valid_error.data[0]))
        exit_str = EarlyStop.EXIT_FLAG_STR.get(stop_flag)
        print('Stopped learning: {}'.format(exit_str))

        pred_train_list = pp.revert(pred_train).data.tolist()
        pred_valid_list = pp.revert(pred_valid).data.tolist()
        animator.add_epoch(epoch, pred_train_list, pred_valid_list)
        animator.set_error_curves(loss_over_epoch, valid_over_epoch)

        final_validation = neural_net.forward(x_valid)
        final_validation_error = mse_loss(final_validation, validation_target_noiseless)

        break

plot_validation_set()
varlist = [hidden_size_2, final_validation_error.mean().data[0]]
save_to_csv('data/hidden_size_2_valid_error.csv', varlist)
#plot_valid_dma()

animator.save_animation('anim.mp4')

# plt.figure()
# plt.plot(loss_over_epoch)
# plt.plot(valid_over_epoch)
#plt.show()
