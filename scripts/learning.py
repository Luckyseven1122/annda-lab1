#!/usr/bin/env python3

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def perceptron_learning(data, labels, epochs=10):

    eta = 0.2

    step_func = np.vectorize(lambda x: int(x >= 0))

    input_size = data.shape[0] + 1
    weights = np.squeeze(rnd.rand(1, input_size))

    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))

    for epoch in range(epochs):
        outputs_from_weights = np.dot(weights, training_data)
        outputs = step_func(outputs_from_weights)
        errors = labels - outputs
        print('Epoch {}, error: {}'.format(epoch, np.sum(errors)))
        plot_boundary(data, labels, weights)
        weights += np.sum(eta*errors)

    return weights


def plot_boundary(data, labels, weights):
    label_color_map = {
        1: 'b',
        0: 'r'
    }
    label_colors = list(map(lambda x: label_color_map.get(x), labels))

    boundary_func = lambda x: -1*(weights[2] + weights[0]*x)/weights[1]
    boundary_x = np.arange(-2, 2)
    boundary_y = np.vectorize(boundary_func)(boundary_x)

    plt.scatter(data[0, :], data[1, :], c=label_colors)
    plt.plot(boundary_x, boundary_y)
    plt.show()

if __name__ == '__main__':

    from datageneration import lin_sep

    data, labels = lin_sep((1, 0), 100)
    weights = perceptron_learning(data, labels)
