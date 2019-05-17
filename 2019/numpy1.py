# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:28:07 2019

@author: du
"""

#numpy already installed with anaconda
import numpy
numpy.__version__

#1.14.3 version

#import numpy as np for easy use
import numpy as np

#built in documentation
#np.<TAB>
np.abs
np?

# --- Bookmark 1     This will appear in the outline explorer

#no declaration of data type required in python
result = 0
for i in range(100):
    result +=1
    print(result)
#select all 3 the lines and then execute

#when assigning more info store in the variable result
    
L = list(range(50))
print(L)
type(L)
type(L[0])
#strings
L2 = [ str(c) for c in L]
L2
#numbers as strings
type(L2)
type(L2[0])
#mixed list
L3 = [True, 2 , [3,4]]
[type(L3)]
[type(item) for item in L3]

#array
import array
L = list(range(10))
A = array.array('i', L)
print(L, A)
A

#arrays from lists
np.array([1,4,5,3])
#same dataype; list can have mixed
np.array([1.5, 3, 4, 7.6])
type(np.array([1.5, 3, 4, 7.6]))
#specify type
np.array([1.5, 3, 4, 7.6], dtype='float32')
np.array([1.5, 3, 4, 7.6], dtype='int') #truncate
#multi-dim array
np.array([range(i, i + 3) for i in [2,4,6]])

#arrays from scratch
np.zeros(10, dtype=int)
np.zeros((3,4), dtype=int)
np.full((3,4), 3.14, dtype=int)
np.full((3,4), 3.14, dtype='float32')
np.full((3,4), 3.14)
np.arange(0,20,2)  #start, step, end
np.linspace(0,1,5)# 5 divisions betw 0 & 1
np.random.random((3,3))
np.random.normal(0,1,(3,4))  #3rows x 4 columns


#basics of npy arrays
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size=6)
x1
x2 = np.random.randint(10, size=(3,4))
x2
x3 = np.random.randint(10, size=(3,4,5))
x3 #4 matrix, 4 rows, 5 col
print('x1 dim - ', x1.ndim)
print('x2 dim - ', x2.ndim)
print('x3 dim - ', x3.ndim)
print('x3 shape - ', x3.shape)
print('x3 size - ', x3.size)
print('x3 dtype - ', x3.dtype)
print('x3 itemsize in bytes - ', x3.itemsize)
print('x3 bytes - ', x3.nbytes)
4 * 3 * 4 * 5  # 4 bytes * no of elements

#array indexing
#x[start:stop:step]
#1-dim
x= np.arange(10)
x
x[:5]  #first 5
x[3:10] # from 3 to 10
x[3:10:2] #alternate from 3 to 10
x[::2]  #alternate from all
x[2:] #start from 2
x[::-1]  #reverse order
x[8:2:-2]  #reverse order from 8 to 2, alternate
x[5::-1]  #reverse order from 5

#2-dim
np.random.seed(0)
x2 = np.random.randint(10, size=(5,4))
x2
x2[0] #first row; start from 0
x2[1] #2nd row
x2[1:3] #2nd to 3rd row
x2[-2] #2nd last row
x2[, :2]  #error
x2[:, [1, 3]] #all rows, 2nd & 4th column
#data[:, ['Column Name1','Column Name2']]
x2[:,[0]]  #1st column
x2[:,[3,2]] #change the order of display of cols
x2
x2[::-1, ]  #rev row
x2[:,::-1 ]  #reverse col
x2[::-1,::-1 ]  #reverse row & col
x2[0]

#subarray - no copy views
#when we change sub array main array is also changed
x2_sub = x2[:2, :2]
x2_sub
x2
x2_sub[0,0] = 99
x2_sub
x2 #changed at both the places

#subarray - copy views 
x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy
x2_sub_copy[0,0] = 9669
x2_sub_copy
x2  #not changed here


#reshaping arrays
grid = np.arange(1,10)  #1 to 9
np.arange?
grid
grid = np.arange(1,10).reshape(3,3)
grid
#1dim to 3dim
x = np.array([1,2,3])
x
x.reshape(1,3)  #row vector
x[np.newaxis, :] #another way for row vector
x.reshape(3,1)  #row to column vector
x[:, np.newaxis] #another way for row vector
#!----
#array concatenation



x1[4]
x1[-1] #last
x1[-2]  #2nd last
x2 = np.random.randint(10, size=(3,4))
x2
x2[1,2]
x2[2,]  #2nd row
x2[2]  #2nd row
x2[[2]] #2nd row
x2[1:3]  #2nd to 3 row
x2[0:3:1]  #all rows
x2[0:3:2]  #alternate eows
x2

#array concatenation
x = np.array([1,2,3])
x
y = np.array([4,5,6])
y
np.concatenate([x,y])  #1dim
np.concatenate([[x],[y]])  #2dim
z= np.array([7,8,9])
z
np.concatenate([x,y,z])  #1dim
np.concatenate([[x],[y],[z]])  #2dim
x2 = np.array([[10,11,12],[13,14,15]])
x2
np.concatenate([x2,x2])
np.concatenate([x2,x2], axis=1)
np.concatenate([x2,x2], axis=0)

#stack commands
x
x2
np.vstack([x,x2])  #row bind
np.hstack([x2,x2])  #column bind
np.dstack([x2,x2])  #3rd dim

#splitting of arrays
xv = np.vstack([x2,x2])  #column bind
xv
upper, lower = np.vsplit(xv, [2])  #split row
upper  #x2
lower #x2
left, right = np.hsplit(xv, [2])  #split row
left
right  #2nd position split, right has only 1 column


#loops are slow-> vectorised operations
#Ufuncs
x = np.arange(4)
x
x * 4
x / 4
#wrappers
np.add(x,4)
np.multiply(x,4)
#others
abs(x)
np.absolute(x)
np.abs(x)
theta = np.linspace(0, np.pi,3)
np.sin(theta)
np.power(x,2)
#special functions
from scipy import special
special.gamma(x)
special.erf(x)

#advanced
x=np.arange(5)
y=np.empty(5)
np.multiply(x, 10, out=y)
y
z = np.zeros(10)
np.power(2,x, out=z[::2])
z
#alternate position square values
x=np.array([1,5,3,2,9])
x
np.add.reduce(x)
#sum values
np.multiply.reduce(x)
np.add.accumulate(x)  #cumulative total
np.multiply.accumulate(x)  #cumulative multiply
np.multiply.outer(x,x)

#Statistical aggregrations
import numpy as np
numpy.random.seed(seed=10)
L = np.random.random(100)
L
sum(L)
np.sum(L)  #faster
min(L), max(L)
np.min(L), np.max(L)
%timeit min(L)
%timeit np.min(L)  #less time
#multi dim
M = np.random.random((3,4))
M
M.sum()
sum(M)  #columnwise
M.sum(axis=0)  #col
M.sum(axis=1) #row
#other functions
#np + sum, prod, mean, std, var, min, max, argmin, argmax, median
#np + percentile, any, all
heights = np.random.randint(100,180,size=50)
heights
heights.min()
heights.max()
heights.mean()
heights.std()
heights.min()
#start with np
np.percentile(heights, (25,70))
np.median(heights)
# --- Plot
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
#run these lines together till y label
plt.hist(heights)
plt.title('Histogram of heights')
plt.xlabel('Height in cm')
plt.ylabel('Count')

#broadcast -----
M=np.ones((3,3))
M
a=np.array([1,2,3])
a+5
M+a
a=np.arange(3)
a
b= np.arange(3)[:, np.newaxis]
b
a + b
#study rules of broadcasting
#rows and columns differ, how to they do arithmetic
np.shape(M)
N = np.ones((3,2))
N
N.shape = (2,3)  #change shape
N

#boolean functions
x = np.array([1,2,3,4,5])
x < 4
x == 3
x != 3
(2 * x ) == (x * 2)
#using Ufuncs
rng = np.random.RandomState(0)
y = rng.randint(5,size=(3,4))
y
y > 3
#counting
np.sum(y > 3)  #3 values > 3
np.count_nonzero(y > 2)
y
np.sum(y > 3, axis=1)
y
np.any(y > 3)
np.all(y > 3)
np.all(y > 3, axis=0)
np.any(y > 3, axis=1)

#joining
np.any((y > 3) & (y > 10))
np.any((y > 3) & (y > 2))
np.any((y > 3) | (y > 10))
np.sum((y > 3) | (y > 10))


#Fancy Indexing
#https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html

#passing array of indices instead 
import numpy as np
rand = np.random.RandomState(0)
x = rand.randint(100,size=10)
x
x[1]
[x[1],x[12],x[82]]
ind = np.array([1,12,82])
ind
x[ind]  #not working
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 
# Index values can be negative. 
arr = x[np.array([1, 3, -3])] 
arr

#modifying values
x = np.arange(10)
x
x[2]
x[2] =99
x
i = [2,5,6]
x[i]
x[i] = 101
x

x = np.zeros(10)
x
np.add.at(x, i, 1)
x

np.random.seed(100)
x = np.random.randn(100)
x
bins = np.linspace(-5,5,20)
bins
counts = np.zeros_like(bins)
counts
i = np.searchsorted(bins, x)
i
np.add.at(counts, i, 1)
counts
plt.plot(bins, counts, linestyle='steps')
plt.hist(x, bins, histtype='step')


#sorting----
x = np.array([5,2, 6, 9,1])
x
np.sort(x)
x.sort()  #sort inplace
x
x = np.array([5,2, 6, 9,1])
x
np.argsort(x)  #indices values
#sorting along rows and columns

rand = np.random.RandomState(42)
x = rand.randint(0,10, (4,6))
x  #matrix of 4 x 6 from 0 to 10
np.sort(x)  #sort rowwise
np.sort(x, axis=1)  #rowwise
np.sort(x, axis=0)  #columnwise

#partition into division
x = np.array([7,2,3,1,6,5,4])
np.partition(x, 3)
np.partition(x, 4)

rand = np.random.RandomState(42)
X = rand.randint(0,10, (4,6))
X
np.partition(X, 3, axis=1)  #rowwise
np.partition(X, 2, axis=0)  #colwise
