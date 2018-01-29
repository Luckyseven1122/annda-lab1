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
        weights += eta*np.sum(errors)

    return weights

def delta_rule_learning(data, labels, epochs=500):

    error = []

    eta = 0.1/data.shape[1]

    input_size = data.shape[0] + 1
    weights = np.squeeze(rnd.rand(1, input_size))

    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))

    for epoch in range(epochs):
        outputs_from_weights = np.dot(weights, training_data)
        errors = labels - outputs_from_weights

        error.append(np.mean(errors**2/2))
        print('Epoch {}, error: {}'.format(epoch, error[epoch]))

        weight_update = np.dot(training_data, errors.T)
        weights += eta*weight_update

    return weights, error


def plot_boundary(data, labels, weights):
    label_color_map = {
        1: 'b',
        0: 'r',
        -1: 'r'
    }
    label_colors = list(map(lambda x: label_color_map.get(x), labels))

    boundary_func = lambda x: -(weights[0]*x + weights[2])/weights[1]
    boundary_x = [-1, 0, 1]
    boundary_y = list(map(boundary_func, boundary_x))

    n_grid = 100
    classify = np.vectorize(lambda x, y: np.sign(weights[0]*x + weights[1]*y + weights[2]))
    grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, n_grid), np.linspace(-2, 2, n_grid))
    grid_class = classify(grid_x, grid_y).flatten()
    grid_colors = np.vectorize(lambda x: label_color_map.get(x))(grid_class)

    plt.scatter(grid_x, grid_y, c=grid_colors, alpha=0.2)
    plt.scatter(data[0, :], data[1, :], c=label_colors)
    plt.plot(boundary_x, boundary_y)
    plt.quiver(boundary_x[1], boundary_y[1], weights[0], weights[1], angles='xy', minlength=1)

if __name__ == '__main__':

    from datageneration import lin_sep

    data, labels = lin_sep((1, 0), 100)
    weights, error = delta_rule_learning(data, labels)


    plt.figure()
    plt.plot(error)
    plt.figure()
    plot_boundary(data, labels, weights)
    plt.show()
