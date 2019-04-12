# -*- coding: utf-8 -*-
"""
NP.partition
"""
import numpy as np
a = np.array([5,2,3,4,1])
np.partition(a,1)
np.partition(a,kth=4)
np.partition(a,(4,2))


x = np.array([1,2,3,4])
x.ndim
x.shape
x[np.newaxis,:]
x[np.newaxis]
y = np.expand_dims(x, axis=0)
y
y.shape
y.ndim

y = np.expand_dims(x, axis=1)  # Equivalent to x[:,np.newaxis]
y
x[:, None]


#squeeze
x = np.array([[[0], [1], [2]]])
x.shape #(1, 3, 1)
x1= np.squeeze(x)
x1
x1.shape #(3,)

x2= np.squeeze(x, axis=0)
x2
x2.shape #(3, 1)

x3= np.squeeze(x, axis=1)  #error
x3.shape

