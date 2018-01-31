#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import itertools

h = []
l = []
e = []
d = []

with open('data/validation_error.csv') as data_file:
    reader = csv.reader(data_file)
    for row in reader:
        d.append((float(row[0]), float(row[1]), float(row[2])))
        h.append(float(row[0]))
        l.append(float(row[1]))
        e.append(float(row[2]))

unique_h = list(set(h))
unique_l = list(set(l))
unique_l.sort()

hl_grid = np.zeros((len(unique_h), len(unique_l)))

for _h, _l in itertools.product(unique_h, unique_l):
    for h_, l_, e_ in d:
        if h_ == _h and l_ == _l:
            curr_e = e_
            break
    h_i = unique_h.index(_h)
    l_i = unique_l.index(_l)
    hl_grid[h_i, l_i] = curr_e

#plt.pcolormesh(unique_h, unique_l, hl_grid.T)
#plt.colorbar()

msh_h, msh_l = np.meshgrid(unique_h, unique_l)
msh_e = np.zeros(msh_h.shape)
for i, (__h, __l) in enumerate(zip(msh_h, msh_l)):
    for j, (_h, _l) in enumerate(zip(__h, __l)):
        for h_, l_, e_ in d:
            if h_ == _h and l_ == _l:
                ee = e_
                break
        msh_e[i, j] = ee

print(msh_h.shape)
print(msh_l.shape)
print(msh_e.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(msh_h, msh_l, hl_grid.T)
ax.set_xlabel('Hidden layer size')
ax.set_ylabel('Lambda regularization weight')
ax.set_zlabel('Validation error')
plt.savefig('data/size_regularization_error.png')
#fig.colorbar()

plt.show()
