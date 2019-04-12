# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:15:06 2018
Dhiraj
@author: du
"""
import numpy as np
x = array([[ 0,  1,  2],[ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11]])
x.sum(1)
x.sum(-1)

rows = (x.sum(-1) % 2) == 0
rows
rows.nonzero()
rows = rows.nonzero()[0]
rows
columns = [0, 2]
x[np.ix_(rows, columns)]
x[rows[:, np.newaxis], columns]
#
x1 = np.arange(1,10).reshape(3,3)
x1
x1_new = x1[:,np.newaxis]
x1_new
x1.shape, x1_new.shape

x2 = np.arange(11,20).reshape(3,3)
x2
x1_new+x2

# expand
x = np.array([1,2])
x.shape
x
#The following is equivalent to x[np.newaxis,:] or x[np.newaxis]:

y = np.expand_dims(x, axis=0)
x, y  # array([1, 2]) array([[1, 2]])
y
y.shape #(1, 2)
y = np.expand_dims(x, axis=1)  # Equivalent to x[:,np.newaxis]
>>> y
array([[1],
       [2]])
>>> y.shape
(2, 1)
Note that some examples may use None instead of np.newaxis. These are the same objects:

>>> np.newaxis is None
True