#!/usr/bin/env python3

"""datageneration.py

Module containing methods for generating datasets.
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def lin_sep(size):
    """Generates two classes of linearly separable data"""

    class1_mean = [0, 0]
    class1_cov = np.diag([0.1, 0.2])
    class1_labels = np.ones(size)

    class1_xy = rnd.multivariate_normal(class1_mean, class1_cov, size).T
    class1_xy_labels = np.vstack((class1_xy, class1_labels))

    class2_mean = [-1.5, -1.5]
    class2_cov = np.diag([0.2, 0.1])
    class2_labels = -1 * np.ones(size)

    class2_xy = rnd.multivariate_normal(class2_mean, class2_cov, size).T
    class2_xy_labels = np.vstack((class2_xy, class2_labels))

    xy_labels = np.hstack((class1_xy_labels, class2_xy_labels))
    rnd.shuffle(xy_labels.T)
    xy = xy_labels[0:2, :]
    labels = xy_labels[-1, :]

    return xy, labels


if __name__ == '__main__':
    data, labels = lin_sep(100)

    label_color_map = {
        1: 'b',
        -1: 'r'
    }
    label_colors = list(map(lambda x: label_color_map.get(x), labels))

    plt.scatter(data[0, :], data[1, :], c=label_colors)
    plt.show()
