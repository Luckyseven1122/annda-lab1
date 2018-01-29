#!/usr/bin/env python3

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def perceptron_learning(data, labels, epochs=100):

    error = []

    eta = 0.1/data.shape[1]

    step_func = np.vectorize(lambda x: int(x >= 0))

    input_size = data.shape[0] + 1
    weights = np.squeeze(rnd.rand(1, input_size))

    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))

    for epoch in range(epochs):
        outputs_from_weights = np.dot(weights, training_data)
        outputs = step_func(outputs_from_weights)
        errors = labels - outputs

        error.append(np. mean(errors**2/2))
        print('Epoch {}, error: {}'.format(epoch, error[epoch]))

        weight_update = np.dot(training_data, errors.T)
        weights += eta*weight_update

    return weights, error

def delta_rule_learning(data, labels, epochs=10):

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

def multilayer_backprop(data, labels, hidden_size, epochs=10):

    error = []

    eta = 0.1#/data.shape[1]

    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))

    hidden_weights = np.squeeze(rnd.rand(hidden_size, data.shape[0] + 1))
    output_weights = np.squeeze(rnd.rand(labels.shape[0], hidden_size + 1))

    sigmoid = lambda x: 2/(1 + np.exp(-x)) - 1
    sigmoid_derivative = lambda x: (1 + x)*(1 - x)/2
    activation_func = np.vectorize(sigmoid)
    activation_derivative = np.vectorize(sigmoid_derivative)

    for epoch in range(epochs):
        hidden_output = activation_func(np.dot(hidden_weights, training_data))
        hidden_output_bias = np.vstack((hidden_output, bias_inputs))
        output = activation_func(np.dot(output_weights, hidden_output_bias))

        errors = labels - output
        error.append(np.mean(errors**2/2))
        print('Epoch: {}, error: {}'.format(epoch, error[epoch]))

        delta_output = (output - labels) * activation_derivative(output)
        error_backprop = np.dot(output_weights.T, delta_output)
        error_backprop = error_backprop[0:hidden_size, :]
        delta_hidden = error_backprop * activation_derivative(hidden_output)

        output_weights -= eta*np.dot(delta_output, hidden_output_bias.T)
        hidden_weights -= eta*np.dot(delta_hidden, training_data.T)

    return hidden_weights, output_weights, error


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
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, n_grid), np.linspace(-3, 3, n_grid))
    grid_class = classify(grid_x, grid_y).flatten()
    grid_colors = np.vectorize(lambda x: label_color_map.get(x))(grid_class)

    plt.scatter(grid_x, grid_y, c=grid_colors, alpha=0.05)
    plt.scatter(data[0, :], data[1, :], c=label_colors)
    plt.plot(boundary_x, boundary_y, c='black')
    plt.quiver(boundary_x[1], boundary_y[1], weights[0], weights[1], angles='xy', minlength=1)

def create_model(weights):

    def func(x):
        sigmoid = lambda x: 2/(1 + np.exp(-x)) - 1
        return sigmoid(np.dot(weights, np.append(x, [1])))

    return func

if __name__ == '__main__':

    from datageneration import lin_sep

    data, labels = lin_sep((1, -1), 100)
    weights, error = delta_rule_learning(data, labels)


    plt.figure()
    plt.plot(error)
    plt.figure()
    plot_boundary(data, labels, weights)
    plt.show()
