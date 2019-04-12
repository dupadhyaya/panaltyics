# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:28:05 2018
Dhiraj
@author: du
"""
import numpy as np
a = np.array([[1,2],[3,4]])
a
a[0,0]
a[0,1]
for i in range(2):
    for  j in range(2):
        print(i, " ", j, " ", " = ",a[i][j] )
a.sum()         #1+2+3+4 = 10
a.sum(axis=None)
a.sum(axis=0)   # array([4, 6])   #1+3 = 4,   2+4 = 6
a.sum(axis=1)   #array([3, 7])    #1+2 = 3,   3+4 = 7
a.sum(axis=2)   # error


#https://stackoverflow.com/questions/19389910/in-python-numpy-what-is-a-dimension-and-axis
a = np.arange(9)
a
a.ndim  # 1
a.shape   #(9,)

b = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
b
b.ndim   #2
b.shape  #(4, 3)
np.sum(b, axis=0)
np.sum(b, axis=1)
np.sum(b, axis=2)

c = np.array([[[0,0,0],[1,1,1]],[[2,2,2],[3,3,3]]])
c
c.ndim
c.shape
c.sum()
c.sum(axis=0)  #3 x 2
c.sum(axis=1)  #  3 x 1
c.sum(axis=2)  # 2 X 2
c

d = np.random.rand(2,2,3,4)
d
d.shape
d.ndim
