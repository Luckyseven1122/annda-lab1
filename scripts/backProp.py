#!/usr/bin/env python3

import numpy as np

def backProp(data, labels, noHidden, epochs=20):
    input_size = data.shape[0] + 1
    eta = 0.1/data.shape[1]
    bias_inputs = np.ones(data.shape[1])
    training_data = np.vstack((data, bias_inputs))
    print(training_data.shape)
    W = np.squeeze(np.random.rand(noHidden, input_size))
    V = np.squeeze(np.random.rand(1, noHidden+1))
    print ("W {}".format(W.shape))
    print ("V {}".format(V.shape))
    dW = 0
    dV = 0
    for epoch in range(epochs):
        #Forward pass
        hin = (np.dot(W, training_data))
        hout = np.divide(2, 1+np.exp(-hin))-1
        hout = np.vstack((hout, bias_inputs))
        oin = np.dot(V, hout)
        out = np.divide(2, 1+np.exp(-oin))-1

        #Backward pass
        delta_o = np.multiply(out - labels, np.multiply(1+out, 1-out))*0.5
        delta_o = delta_o.reshape(1, -1)
        V = V.reshape(1,-1)
        delta_h = np.multiply(np.dot(V.T, delta_o), np.multiply(1+hout, 1-hout))*0.5
        delta_h = delta_h[0:noHidden, :]
        print(delta_h.shape)

        #Weight update
        alpha = 0.9
        dW = np.multiply(dW, alpha) - np.multiply(np.dot(delta_h, training_data.T), 1-alpha)
        dV = np.multiply(dV, alpha) - np.multiply(np.dot(delta_o, hout.T), 1-alpha)
        W = W + np.multiply(dW, eta)
        V = V + np.multiply(dV, eta)


if __name__ == '__main__':

    from datageneration import lin_sep

    data, labels = lin_sep((1, 0), 5)
    noHidden = 4
    weights, error = backProp(data, labels, noHidden)
