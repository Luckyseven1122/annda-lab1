#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def trainMultiLayer(data, labels, eta, noHidden, nOutput=1, epochs=100):
    input_size = data.shape[0] + 1
    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))

    W = np.squeeze(np.random.rand(noHidden, input_size))
    V = np.squeeze(np.random.rand(nOutput, noHidden+1))
    V = V.reshape(1, -1)
    error = []
    dW = 0
    dV = 0
    alpha = 0.9

    for epoch in range(epochs):
        #Forward pass
        hin = np.dot(W, training_data)
        hout = np.divide(2, 1+np.exp(-hin)) - 1
        hout = np.vstack((hout, bias_inputs))
        oin = np.dot(V, hout)
        out = np.divide(2, 1+np.exp(-oin)) - 1
        errors = labels - out
        error.append(np.sum(np.power(errors, 2)/2)/len(errors[0]))
        print("Epoch: {}, Error: {}".format(epoch, error[epoch]))

        #Backward pass
        delta_o = np.multiply(out-labels, np.multiply(1+out, 1-out)*0.5)
        delta_o = delta_o.reshape(1, -1)
        delta_h = np.multiply(np.dot(V.T, delta_o), np.multiply(1+hout, 1-hout)*0.5)
        delta_h = delta_h[0:noHidden, :]

        #Weight update
        dW = (dW*alpha) - np.dot(delta_h, training_data.T) * (1-alpha)
        dV = (dV*alpha) - np.dot(delta_o, hout.T) * (1-alpha)
        W = W + dW*eta
        V = V + dV*eta
    return out, W, V, error

def evaluateMultiLayer(W, V, testData, labels):
    bias_inputs = np.ones(testData.shape[1])
    testData = np.vstack((testData, bias_inputs))
    hin = np.dot(W, testData)
    hout = np.divide(2, 1+np.exp(-hin)) - 1
    hout = np.vstack((hout, bias_inputs))
    oin = np.dot(V, hout)
    out = np.divide(2, 1+np.exp(-oin)) - 1
    errors = labels - out
    error = np.sum(np.power(errors, 2)/2)/len(errors[0])
    return error

if __name__ == '__main__':
    from datageneration import lin_sep
    from learning import plot_boundary
    data, labels = lin_sep((1, 0), 100)
    testData, testLabels = lin_sep((1, 0), 50)
    noHidden = 2
    eta = 0.01
    epochs = 20
    nOutput = 1
    W, V, error = trainMultiLayer(data, labels, eta, noHidden, nOutput, epochs)
    testError = evaluateMultiLayer(W, V, testData, testLabels)
    print("Test error: {}".format(testError))

    label_color_map = {
        1: 'b',
        0: 'r',
        -1: 'r'
    }
    label_colors = list(map(lambda x: label_color_map.get(x), labels))
    plt.figure()
    plt.scatter(data[0, :], data[1, :], c=label_colors)
    plt.figure()
    plt.scatter(testData[0, :], testData[1, :], c=label_colors)
    plt.show()
