#!/usr/bin/env python3

import itertools
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datageneration import mackey_glass

class EarlyStop:

    MOVING_AVERAGE_RANGE = 25

    INCREASING_THRESHOLD = 1e-3
    INCREASING_MAX_ITER = 300

    STOPPED_LEARNING_THRESHOLD = 1e-6
    STOPPED_LEARNING_MAX_ITER = 300

    NORMAL_EXIT_FLAG = 0
    STOPPED_LEARNING_EXIT_FLAG = -1
    INCREASING_EXIT_FLAG = -2
    BREAK_FLAGS = (
        STOPPED_LEARNING_EXIT_FLAG,
        INCREASING_EXIT_FLAG
    )

    EXIT_FLAG_STR = {
        STOPPED_LEARNING_EXIT_FLAG: 'Semi stable validation error',
        INCREASING_EXIT_FLAG: 'Increasing validation error'
    }

    def __init__(self):
        self.increasing_counter = 0
        self.stopped_learning_counter = 0
        self.global_min = float('inf')

        init_moving_average = [float('inf')]*self.MOVING_AVERAGE_RANGE
        self.prev_x = collections.deque(init_moving_average)
        self.ma_prev = float('inf')
        self.dma = 0

    def diff_moving_average(self, new_val):
        self.prev_x.popleft()
        self.prev_x.append(new_val)
        new_ma = np.mean(self.prev_x)
        dma = self.ma_prev - new_ma
        self.ma_prev = new_ma
        return dma

    def check_for_early_stop(self, new_val):

        set_new_global_min = self.check_global_min(new_val)
        dma = self.diff_moving_average(new_val)
        self.dma = dma
        increasing_raised = self.check_increasing(dma)
        stopped_learning_raised = self.check_stopped_learning(dma)

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

    def check_increasing(self, dma):

        if dma >= self.INCREASING_THRESHOLD:
            self.increasing_counter += 1
        else:
            self.increasing_counter = 0

        if self.increasing_counter >= self.INCREASING_MAX_ITER:
            return True
        return False

    def check_stopped_learning(self, dma):

        if abs(dma) <= self.STOPPED_LEARNING_THRESHOLD:
            self.stopped_learning_counter += 1
        else:
            self.stopped_learning_counter = 0

        if self.stopped_learning_counter >= self.STOPPED_LEARNING_MAX_ITER:
            return True
        return False


def preprocess_data(data):
    mean = np.mean(data)
    std = np.std(data)
    processed_data = (data - mean)/std
    return processed_data, mean, std

def process_data(data, mean, std):
    return data*std + mean

neural_net = nn.Sequential(
    nn.Linear(5, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1)
)

mglass = mackey_glass(1.5, 3000)
mglass, mean, std = preprocess_data(mglass)

t_train = np.arange(300, 800)
t_valid = np.arange(801, 1300)
t_test = np.arange(1301, 1500)
training_data = torch.t(Variable(torch.Tensor([
    mglass[t_train-20],
    mglass[t_train-15],
    mglass[t_train-10],
    mglass[t_train-5],
    mglass[t_train]
])))
validation_data = torch.t(Variable(torch.Tensor([
    mglass[t_valid-20],
    mglass[t_valid-15],
    mglass[t_valid-10],
    mglass[t_valid-5],
    mglass[t_valid]
])))

training_target = Variable(torch.Tensor(mglass[t_train+5]))
validation_target = Variable(torch.Tensor(mglass[t_valid+5]))

loss_fnc = nn.MSELoss()

learning_rate = 0.7
alpha = 0.4

loss_over_epoch = []
valid_over_epoch = []

training_predictions = []
validation_predictions = []
animated_epochs = []

curr_grads = []
optimal_params = []
for param in neural_net.parameters():
    curr_grads.append(0)
    optimal_params.append(param)
    param.grad = Variable(torch.zeros(param.size()))

prev_valid_error = float('inf')
valid_min = float('inf')
stopped_learning_counter = 0

early_stopper = EarlyStop()

for epoch in itertools.count():
    # errors over training data
    prediction = neural_net.forward(training_data)
    loss = loss_fnc(prediction, training_target)
    loss_over_epoch.append(loss.data[0])

    # errors over validation data
    validation_prediction = neural_net.forward(validation_data)
    valid_error = loss_fnc(validation_prediction, validation_target)
    valid_over_epoch.append(valid_error.data[0])

    for param, curr_grad in zip(neural_net.parameters(), curr_grads):
        curr_grad = param.grad.data * alpha

    neural_net.zero_grad()

    loss.backward()
    for param, curr_grad in zip(neural_net.parameters(), curr_grads):
        param.data -= curr_grad + (learning_rate * param.grad.data)*(1 - alpha)

    if epoch % 25 == 0:
        print('Epoch {:4d}, loss: {:6.5f}, validation error: {:6.5f}'.format(
               epoch, loss.data[0], valid_error.data[0]))

        training_predictions.append(process_data(prediction, mean, std).data.tolist())
        validation_predictions.append(process_data(validation_prediction, mean, std).data.tolist())
        animated_epochs.append(epoch)

    set_global_min, stop_flag = early_stopper.check_for_early_stop(valid_error.data[0])

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

        training_predictions.append(process_data(prediction, mean, std).data.tolist())
        validation_predictions.append(process_data(validation_prediction, mean, std).data.tolist())
        animated_epochs.append(epoch)
        break


def plot_pred(frame, figure, ax_train, ax_valid, ax_train_error, ax_valid_error):
    try:
        [a.lines.pop(1) for a in [ax_train, ax_valid, ax_train_error, ax_valid_error]]
    except IndexError:
        pass
    curr_epoch_line_x = (animated_epochs[frame], animated_epochs[frame])
    curr_epoch_line_y = (0, 1)
    ax_train.plot(t_train + 5, training_predictions[frame], c='#ff5050')
    ax_valid.plot(t_valid + 5, validation_predictions[frame], c='#990000')
    ax_train_error.plot(curr_epoch_line_x, curr_epoch_line_y, c='black')
    ax_valid_error.plot(curr_epoch_line_x, curr_epoch_line_y, c='black')
    figure.suptitle('Epoch {}'.format(animated_epochs[frame]))

f = plt.figure()
a = f.add_subplot(111)
a.set_ylim(0, 0.2)
plt.plot(loss_over_epoch)
plt.plot(valid_over_epoch)

anim_len = 10
fps = len(training_predictions)/anim_len

fig = plt.figure(figsize=(10, 6))
ax_train = fig.add_subplot(221)
ax_valid = fig.add_subplot(222)
ax_train_error = fig.add_subplot(223)
ax_valid_error = fig.add_subplot(224)
ax_train.set_ylim(0.2, 1.4)
ax_valid.set_ylim(0.2, 1.4)
ax_train.set_xlim(t_train[0], t_train[-1])
ax_valid.set_xlim(t_valid[0], t_valid[-1])
ax_train.set_title('Training set')
ax_valid.set_title('Validation set')
ax_train.plot(t_train+5, process_data(training_target, mean, std).data.tolist(), c='#6699ff')
ax_valid.plot(t_valid+5, process_data(validation_target, mean, std).data.tolist(), c='#000099')

ax_train_error.set_ylim(0, 0.2)
ax_valid_error.set_ylim(0, 0.2)
ax_train_error.set_xlim(0, len(loss_over_epoch))
ax_valid_error.set_xlim(0, len(valid_over_epoch))
ax_train_error.plot(loss_over_epoch, c='black')
ax_valid_error.plot(valid_over_epoch, c='black')
fargs_tup = (fig, ax_train, ax_valid, ax_train_error, ax_valid_error)
an = animation.FuncAnimation(fig, plot_pred, fargs=fargs_tup,
                             interval=1, frames=len(training_predictions))
mpeg_writer = animation.FFMpegWriter(fps=fps)
an.save('mackey_glass_prediction.mp4', writer=mpeg_writer)
# plt.figure()
# plt.plot(loss_over_epoch)
# plt.plot(valid_over_epoch)
#plt.show()
