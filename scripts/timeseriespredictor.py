#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datageneration import mackey_glass

def preprocess_data(data):
    mean = np.mean(data)
    std = np.std(data)
    processed_data = (data - mean)/std
    return processed_data, mean, std

def process_data(data, mean, std):
    return data*std + mean

neural_net = nn.Sequential(
    nn.Linear(5, 15),
    nn.Sigmoid(),
    nn.Linear(15, 1)
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

learning_rate = 0.25
alpha = 0.25

loss_over_epoch = []
valid_over_epoch = []

training_predictions = []
validation_predictions = []
animated_epochs = []

curr_grads = []
for param in neural_net.parameters():
    curr_grads.append(0)
    param.grad = Variable(torch.zeros(param.size()))

for epoch in range(5000):
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

    if epoch % 50 == 0:
        print('Epoch {:4d}, loss: {:6.5f}, validation error: {:6.5f}'.format(
               epoch, loss.data[0], valid_error.data[0]))

        training_predictions.append(process_data(prediction, mean, std).data.tolist())
        validation_predictions.append(process_data(validation_prediction, mean, std).data.tolist())
        animated_epochs.append(epoch)

def plot_pred(frame, ax_train, ax_valid):
    try:
        ax_train.lines.pop(1)
        ax_valid.lines.pop(1)
    except IndexError:
        pass
    ax_train.plot(t_train + 5, training_predictions[frame], c='#ff5050')
    ax_valid.plot(t_valid + 5, validation_predictions[frame], c='#990000')
    fig.suptitle('Epoch {}'.format(animated_epochs[frame]))

f = plt.figure()
a = f.add_subplot(111)
a.set_ylim(0, 0.2)
plt.plot(loss_over_epoch)
plt.plot(valid_over_epoch)

fig = plt.figure(figsize=(10, 6))
ax_train = fig.add_subplot(211)
ax_valid = fig.add_subplot(212)
ax_train.set_ylim(0.2, 1.4)
ax_valid.set_ylim(0.2, 1.4)
ax_train.set_xlim(t_train[0], t_train[-1])
ax_valid.set_xlim(t_valid[0], t_valid[-1])
ax_train.set_title('Training set')
ax_valid.set_title('Validation set')
ax_train.plot(t_train+5, process_data(training_target, mean, std).data.tolist(), c='#6699ff')
ax_valid.plot(t_valid+5, process_data(validation_target, mean, std).data.tolist(), c='#000099')
an = animation.FuncAnimation(fig, plot_pred, fargs=(ax_train, ax_valid),
                             interval=1, frames=len(training_predictions))
mpeg_writer = animation.FFMpegWriter(fps=10)
an.save('mackey_glass_prediction.mp4', writer=mpeg_writer)
# plt.figure()
# plt.plot(loss_over_epoch)
# plt.plot(valid_over_epoch)
#plt.show()
