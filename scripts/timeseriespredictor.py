#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datageneration import mackey_glass

neural_net = nn.Sequential(
    nn.Linear(5, 15),
    nn.Sigmoid(),
    nn.Linear(15, 1)
)

mglass = mackey_glass(1.5, 3000)

mglass_mean = np.mean(mglass)

mglass = mglass - mglass_mean

t_train = np.arange(30, 1500)
t_valid = np.arange(1501, 2700)
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
learning_rate = 0.1

loss_over_epoch = []
valid_over_epoch = []

training_predictions = []
validation_predictions = []
animated_epochs = []

for epoch in range(10000):
    prediction = neural_net.forward(training_data)
    loss = loss_fnc(prediction, training_target)
    loss_over_epoch.append(loss.data[0])
    validation_prediction = neural_net.forward(validation_data)
    valid_error = loss_fnc(validation_prediction, validation_target)
    valid_over_epoch.append(valid_error.data[0])
    neural_net.zero_grad()
    loss.backward()
    for param in neural_net.parameters():
        param.data -= learning_rate * param.grad.data

    print('Epoch {:4d}, loss: {:6.5f}, validation error: {:6.5f}'.format(epoch, loss.data[0], valid_error.data[0]))

    if epoch % 50 == 0:
        pred_w_mean = prediction + mglass_mean
        vald_w_mean = validation_prediction + mglass_mean
        training_predictions.append(pred_w_mean.data.tolist())
        validation_predictions.append(vald_w_mean.data.tolist())
        animated_epochs.append(epoch)

def plot_pred(frame, axis):
    try:
        ax.lines.pop(2)
        ax.lines.pop(3)
    except IndexError:
        pass
    plt.plot(t_train + 5, training_predictions[frame], c='#ff5050')
    plt.plot(t_valid + 5, validation_predictions[frame], c='#990000')
    axis.set_title('Epoch {}'.format(animated_epochs[frame]))

f = plt.figure()
a = f.add_subplot(111)
a.set_ylim(0, 0.2)
plt.plot(loss_over_epoch)
plt.plot(valid_over_epoch)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_ylim(0.2, 1.4)
ax.set_xlim(0, 2705)
train_target_w_mean = training_target + mglass_mean
valid_target_w_mean = validation_target + mglass_mean
plt.plot(t_train, train_target_w_mean.data.tolist(), c='#6699ff')
plt.plot(t_valid, valid_target_w_mean.data.tolist(), c='#000099')
an = animation.FuncAnimation(fig, plot_pred, fargs=(ax,), interval=1, frames=len(training_predictions))
mpeg_writer = animation.FFMpegWriter(fps=10)
an.save('mackey_glass_prediction.mp4', writer=mpeg_writer)
# plt.figure()
# plt.plot(loss_over_epoch)
# plt.plot(valid_over_epoch)
#plt.show()
