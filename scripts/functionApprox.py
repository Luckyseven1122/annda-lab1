import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from backProp import trainMultiLayer
from matplotlib import cm
import matplotlib.animation as animation

def calcZ(x, y):
    zx = np.exp(np.multiply(-x, x*0.1))
    zy = np.exp(np.multiply(-y, y*0.1))
    z = np.dot(zx, zy.T) - 0.5
    return z

def generateFunctionData(x, y, z):
    ndata = len(x)*len(y)
    targets = z.reshape((1, ndata))
    xx, yy = np.meshgrid(x, y)
    patternsxx = xx.reshape((1, ndata))
    patternsyy = yy.reshape((1, ndata))
    patterns = np.vstack((patternsxx, patternsyy))
    return targets, xx, yy, patterns

def update_plot(frame_number, zarray, plot, ax, x, y):
    print("Frame: {}".format(frame_number))
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[frame_number], cmap="magma")

def forwardPass(W, V, data):
    hin = np.dot(W, data)
    hout = np.divide(2, 1+np.exp(-hin)) - 1
    hout = np.vstack((hout, bias_inputs))
    oin = np.dot(V, hout)
    out = np.divide(2, 1+np.exp(-oin)) - 1
    return out, hout

def backwardPass(out, hout, V, targets, noHidden):
    delta_o = np.multiply(out-targets, np.multiply(1+out, 1-out)*0.5)
    delta_o = delta_o.reshape(1, -1)
    delta_h = np.multiply(np.dot(V.T, delta_o), np.multiply(1+hout, 1-hout)*0.5)
    delta_h = delta_h[0:noHidden, :]
    return delta_o, delta_h

def updateWeights(delta_o, delta_h, hout, data, eta):
    alpha = 0.9
    dW = (dW*alpha) - np.dot(delta_h, data.T) * (1-alpha)
    dV = (dV*alpha) - np.dot(delta_o, hout.T) * (1-alpha)
    W = W + dW*eta
    V = V + dV*eta
    return V, W

def biasInput(data):
    print(data.shape)
    bias = np.ones(data.shape[1])
    biasedData = np.vstack((data, bias))
    return biasedData

def trainAndVisualise(nEpochs, noHidden, nOutput, eta):
    x = np.arange(-5, 5, 0.5).reshape((-1, 1))
    y = np.arange(-5, 5, 0.5).reshape((-1, 1))
    z = calcZ(x, y)
    targets, xx, yy, patterns = generateFunctionData(x, y, z)
    zzArray = []
    input_size = patterns.shape[0] + 1
    bias_inputs = np.ones(patterns.shape[1])
    training_data = np.vstack((patterns, bias_inputs))

    W = np.squeeze(np.random.rand(noHidden, input_size))
    V = np.squeeze(np.random.rand(nOutput, noHidden+1))
    V = V.reshape(1, -1)
    error = []
    dW = 0
    dV = 0
    alpha = 0.9
    for epoch in range(nEpochs):
        #Forward pass
        hin = np.dot(W, training_data)
        hout = np.divide(2, 1+np.exp(-hin)) - 1
        hout = np.vstack((hout, bias_inputs))
        oin = np.dot(V, hout)
        out = np.divide(2, 1+np.exp(-oin)) - 1
        errors = targets - out
        error.append(np.sum(np.power(errors, 2)/2)/len(errors[0]))
        print("Epoch: {}, Error: {}".format(epoch, error[epoch]))

        #Backward pass
        delta_o = np.multiply(out-targets, np.multiply(1+out, 1-out)*0.5)
        delta_o = delta_o.reshape(1, -1)
        delta_h = np.multiply(np.dot(V.T, delta_o), np.multiply(1+hout, 1-hout)*0.5)
        delta_h = delta_h[0:noHidden, :]

        #Weight update
        dW = (dW*alpha) - np.dot(delta_h, training_data.T) * (1-alpha)
        dV = (dV*alpha) - np.dot(delta_o, hout.T) * (1-alpha)
        W = W + dW*eta
        V = V + dV*eta
        zz = out.reshape((len(x), len(y)))
        zzArray.append(zz)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_surface(x, y, zzArray[0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(-0.7,0.7)
    animate = animation.FuncAnimation(fig, update_plot, nEpochs, interval=20, fargs=(zzArray, plot, ax, x, y))
    #animate.save('./animation.gif', writer='imagemagick', fps=60)
    plt.show()

def doubleForward(n):
    x = np.arange(-5, 5, 0.5).reshape((-1, 1))
    y = np.arange(-5, 5, 0.5).reshape((-1, 1))
    z = calcZ(x, y)
    targets, xx, yy, patterns = generateFunctionData(x, y, z)

    trainingPatterns = biasInput(patterns[:, 0:n])
    trainingTargets = biasInput(targets[0, 0:n])
    patterns = biasInput(patterns)
    targets = biasInput(targets)

    input_size = patterns.shape[0] + 1

    W = np.squeeze(np.random.rand(noHidden, input_size))
    V = np.squeeze(np.random.rand(nOutput, noHidden+1))
    V = V.reshape(1, -1)

    dW = 0
    dV = 0
    for epoch in range(nEpochs):
        out, hout = forwardPass(W, V, patterns)
        outTrain, houtTrain = forwardPass(W, V, trainingPatterns)
        delta_o, delta_h = backwardPass(outTrain, houtTrain, V, trainingPatterns, noHidden)
        V, W = updateWeights(delta_o, delta_h, houtTrain, trainingPatterns, eta)
if __name__ == '__main__':
    doubleForward(1)
