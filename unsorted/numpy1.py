# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:28:57 2017 by Dhiraj Upadhyaya
Introduction to NumPy Numerical Python
"""
import numpy
print(numpy.version.version)

import numpy as np

#np.<TAB> or np?

#Python List - run selected line
L = list(range(10))
print(L)
type(L[0])

###works only in IPython
#L2=[str(c) for c in L]
#print(L2) ###

L3 = [True, '2', 3.0, 4]
print(L3)
print(type(L3))
print([type(item) for item in L3])

#Fixed type Arrays in Python
import array
L=list(range(10))
A=array.array('i',L)  # as integer
A

#Creating arrays from Python Lists
np.array([1,4,2,5,3])  # same datatype
np.array([3.14, 4, 2., 3.])
print(np.array([3.14, 4, 2., 3.]))

np.array([range(i, i+3) for i in [2,4,6]])
#inner lists as rows

#Creating arrays from Scratch Pg 39
np.zeros(10)
np.zeros(10,dtype=int)
np.zeros(shape=(2,3),dtype=float)

np.ones((3,5), dtype=float)
np.full((3,4), 3.14)
np.arange(0,20,2)
np.linspace(0,1,5)
np.random.random((3,3))
np.random.normal(0,1,(3,3)) #sd=1, mean=0, 3x3
np.random.randint(0,10,(3,3))
np.eye(3)
np.empty(3)

#Numpy Standard Data Types Pg 41
np.zeros(10, dtype='int16')
np.zeros(10, dtype=np.int16)

#Basics of NumPy Arrays

#Numypy Array Attributes
import numpy as np
np.random.seed(0)
x1= np.random. randint(10, size=6)
print(x1)
x2= np.random.randint(10, size=(3,4))
x3= np.random.randint(10, size=(3,4,5))
x2
x3
x3.ndim
print(x3.ndim)
x3.shape
x3.size
x3.dtype
x3.nbytes


#Array Indexing : Accessing Single Elements
x1
x1[0] 
x1[4]
x1[-1]
x1[-2]
x2
x2[0]
x2[0,0]
x2[2,0]
x2[2,-1]
x2[0,0]=12  #mutable
x2
x1[0] = 3.14  # truncated
x1

#Array Slicing : Accessing Subarrays
#x[start:Stop:step]
x= np.arange(10)
x
x[:5]
x[2:6]
x[1:-1]
x[4:]
x[::2]
x[1::2]
x[::-1]
x[::-2]
x[5::-2]

#Multi Dimn subarrays
x2
x2[:2, :3]
x2[:3,::2]
x2
x2[::-1, ::-1]

#Accessing array rows & columns
x2[:,0]
x2[0,:]
x2
x2[0]

#subarrays as no copy views
x2
x2_sub = x2[:2,:2]
x2_sub
x2_sub[0,0] = 99
x2

#Copies of arrays
x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy
x2_sub_copy [0,0] = 42
x2
x2_sub_copy

#Reshaping of Arrays


#universal Functions Pg 51
import numpy as np
values = np.random.randint(1,10, size=5)
values
print(1.0/values)
%timeit (1.0/values)

np.arange(5)/ np.arange(1,6)
x = np.arange(9).reshape((3,3))
x
x= np.arange(4)
x
x+5
x-5
x*2
x/2
x//2
-x
x**2
x%2
-(0.5 * x + 1) **2

#wrappers
np.add(x, 2)
np.subtract(x,5)

x = np.array([-2,-1,0,1,2])
x
np.abs(x)
np.absolute(x)

#Trigo 
theta = np.linspace(0, np.pi, 3)
theta
np.sin(theta)
np.cos(theta)
np.tan(theta)

x=[-1,0,1]
x
np.arcsin(x)
np.arccos(x)
np.arctan(x)

#exponents
x=[1,2,3]
x
np.exp(x)
np.exp2(x)
np.power(3,x)
x[1,2,4,10]
x
np.log(x)
np.log2(x)
np.log10(x)


x=[0,0.001, 0.01, 0.1]
