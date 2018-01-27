#!/usr/bin/env python3

"""datageneration.py

Module containing methods for generating datasets.
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def lin_sep(size):
    """Generates two classes of linearly separable data"""

    size_tup = (2, size)

    class1_mean = [1, 1]
    class1_cov = np.diag([0.1, 0.2])

    class1_x, class1_y = rnd.multivariate_normal(class1_mean, class1_cov, size_tup)

    class2_mean = [-1, -1]
    class2_cov = np.diag([0.2, 0.1])

    class2_x, class2_y = rnd.multivariate_normal(class2_mean, class2_cov, size_tup)

    return class1_x, class1_y, class2_x, class2_y


if __name__ == '__main__':
    class1_x, class1_y, class2_x, class2_y = lin_sep(100)
    plt.scatter(class1_x, class1_y, color='r')
    plt.scatter(class2_x, class2_y, color='b')
    plt.show()
