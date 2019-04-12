# -*- coding: utf-8 -*-
"""
NP newaxis
"""
import numpy as np
a = np.array([1,2,3,4])
a
a.shape
a1= a[np.newaxis, :]
a1.shape
a
a2 = a[:, np.newaxis]
a2.shape
a2
a1a= a.reshape(1,4)
a1a.shape
a2a= a.reshape(4,1)
a2a.shape
a2a

np.tile(a,(1,4))
np.tile(a,(4,1))


a3= a2[:, np.newaxis, :]
a3
a3= a1[:, np.newaxis, :]
a3


# randint
x = np.random.randint(8, size=(1,2,4))
x
a = np.random.randint(1,8,8)
a
x = np.array([np.random.randint(1,8,8)]).reshape(1,2,4)
x.reshape(8)
x1=x[:,newaxis,:,:]
x1.shape


x= np.arange(12)
x
x.shape
x1=x[newaxis,:]
x1.shape
x2=x1[newaxis,:]
x2.shape
x[0]
x[0:10]
x3=x2[newaxis, :]
x3.shape
x3
a = np.array([[1, 2, 3], [4, 5, 6]])
a
a.ravel()
a.T
a.T.ravel()
a.reshape((2, -1))
