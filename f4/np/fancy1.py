# -*- coding: utf-8 -*-
"""
Tue Feb  6 09:38:17 2018: Dhiraj
"""

# Fancy Indexing
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100,size=10)
x


#Indexing
[ x[3], x[7], x[2] ]

ind = [3,7,2]
x[ind]

#ind = [[3,7], [2,4]]  # does not work 
ind = np.array([[3,7],[2,4]])
ind.shape, ind.size
x[ind]  #output shape not the shape of the data

X = np.arange(12).reshape((3,4))
row = np.array([0,1,2])
col = np.array([2,1,3])
row, col
X
X[row, col]
row[:, np.newaxis]
X[row[:, np.newaxis], col]   #leave col0

a=np.array([1,4,8,9,5])
a
y=[a[0],a[2],a[4]]
y
ind=[0,2,4]
y=a[ind]
y

a=np.arange(0,80,10)
y=take(a, [0,2,4]]


#combined indexing combine fancy and simple indices:

X[2, [2, 0, 1]]

#combine fancy indexing with slicing:
X
X[1:, [2, 0, 1]]

#combine fancy indexing with masking:
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]


#Selecting Random Points
mean = [0, 0]
cov = [[1, 2],   [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # for plot styling

X.shape[0]
plt.scatter(X[:, 0], X[:, 1])
indices = np.random.choice(X.shape[0], 20, replace=False)
indices
selection = X[indices]  # fancy indexing here
selection
selection.shape
