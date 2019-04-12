# -*- coding: utf-8 -*-
"""
Tue Feb  6 10:32:09 2018: Dhiraj
"""

#Chapter-2

#%% Numpy Array Attributes
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)

print("dtype:", x3.dtype)

print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


#%% Array Indexing: Accessing Single Elements
#1D
x1

x1[0]

x1[4]

#To index from the end of the array, you can use negative indices:
x1[-1]

x1[-2]

#In a multi-dimensional array, items can be accessed using a comma-separated tuple of indices:
x2

x2[0, 0]

x2[2, 0]

x2[2, -1]

#Values can also be modified using any of the above index notation:

x2[0, 0] = 12
x2

x1[0] = 3.14159  # this will be truncated!
x1
#%% Array Slicing: Accessing Subarrays
#x[start:stop:step]

#%% One-dimensional subarrays
x = np.arange(10)
x
x[:5]  # first five elements

x[5:]  # elements after index 5

x[4:7]  # middle sub-array

x[::2]  # every other element

x[1::2]  # every other element, starting at index 1

x[::-1]  # all elements, reversed

x[5::-2]  # reversed every other from index 5

#%% Multi-dimensional subarrays

x2

x2[:2, :3]  # two rows, three columns

x2[:3, ::2]  # all rows, every other column

#Finally, subarray dimensions can even be reversed together:

x2[::-1, ::-1]

#Accessing array rows and columns
#One commonly needed routine is accessing of single rows or columns of an array. This can be done by combining indexing and slicing, using an empty slice marked by a single colon (:):

print(x2[:, 0])  # first column of x2

print(x2[0, :])  # first row of x2

#In the case of row access, the empty slice can be omitted for a more compact syntax:

print(x2[0])  # equivalent to x2[0, :]



#%% Subarrays as no-copy views
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)

x2_sub[0, 0] = 99
print(x2_sub)
print(x2)

#%% Creating copies of arrays
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)

#%% Reshaping of Arrays
grid = np.arange(1, 10).reshape((3, 3))
print(grid)
x = np.array([1, 2, 3])

# row vector via reshape
x.reshape((1, 3))
# row vector via newaxis
x[np.newaxis, :]

# column vector via reshape
x.reshape((3, 1))

# column vector via newaxis
x[:, np.newaxis]

#%%  Average Heights of People
heights = np.random.randint(4,7, 40)
heights
heights = np.random.normal(5,1,40)
heights = np.random.normal(5,.5,40).round(1)
heights
heights.mean(), heights.std(), heights.var()
heights.min(), heights.max(), heights.var()
np.percentile(heights,25)
np.sort(heights)
np.percentile(heights, [0,25,50,75,100])
np.ptp(heights)
np.median(heights)
np.histogram(heights)

#plot
%matplotlib inline
import matplotlib.pyplot as plt
#run together next 4 lines
plt.hist(heights)
plt.title("Height")
plt.xlabel(" Height cm")
plt.ylabel("Numbers")

#%%  Pg 62 Computation on Arrays : Broadcasting
a = np.array([0,1,2])
b = np.array([5,5,5])
a + b
# arrays maybe of different sizes

a + 5

M = np.ones((3,3))
M
M + a
#a is stretched to 3 rows 

a = np.arange(3)
a
b = np.arange(3)[:, np.newaxis]
b
a + b
a.shape, b.shape

#Broadcasting rules
#Rule1  : 1D to 1D
M = np.ones((2,3))
a = np.arange(3)
M,a
M.shape, a.shape
M + a   #a + M
(M+a).shape
M.shape, a.shape

#Broadcasting Example 2
a = np.arange(3).reshape((3,1))
b = np.arange(3)
a,b
a.shape, b.shape
(a + b)
(a+b).shape

#Broadcasting Eg3
M = np.ones((3,2))
a = np.arange(3)
M, a
M.shape, a.shape
M + a

a[:, np.newaxis].shape
M.shape
M + a[:, np.newaxis]
#%% Pg 70 Comparisons, Masks & Boolean Logic

#Example : Counting Rainy Days
import numpy as np
import pandas as pd

# use pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv("pdabook/Seattle2014.csv")["PRCP"].values
#url = "https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/data/Seattle2014.csv"
#rainfall = pd.read_csv(url)
inches = rainfall / 254
inches.shape
inches = rainfall / 254.0  # 1/10mm -> inches
inches.shape
inches[1:10]

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn ; seaborn.set()
plt.hist(inches, 40)

#Comparison Operators as ufuncs
x = np.array([1,2,3,4,5])
x
x<3  # which all x values < 3
x<=3
x==3
x!=3
np.equal(x,3)
np.greater(x,3)

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3,4))
x
x< 6
# Pg 73 Working with Boolean Arrays
x
np.count_nonzero(x < 6)
np.sum(x < 6)
np.sum(x < 6, axis=1)
np.any(x > 8)
np.all(x < 10)
np.all(x < 8, axis=1)
x


#Pg 74 Boolean operators
np.sum((inches > 0.5) & (inches < 1))
np.sum((x > 3) & (x < 8))  # 2 brackets

x < 5
x[x < 5]   # all values 1-D which satisfy the condition

#%% Pg 78 Fancy Indexing


# Combined Indexing

# Selecting Random Points


# Modifying Values with Fancy Indexing

# Binning Data
np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5,5,20)
bins
counts = np.zeros_like(bins)
i = np.searchsorted(bins, x)
i  # into bins
np.add.at(counts, i, 1)
plt.plot(bins, counts, linestyle='steps')

#faster way
plt.hist(x, bins, histtype='step')


#%% Pg 86 - Fast Sorting  in NP
x = np.array([2,1,4,3,5])
np.sort(x) # copy
x
x.sort()  #change in place
x

x = np.array([2,1,4,3,5])
i= np.argsort(x)  #copy
i
x
np.argsort(x)  # copy
x
x[i]  #sorted list using index values

#Sorting along rows or columns
rand = np.random.RandomState(42)
X= rand.randint(0, 10, 24)
X
X= rand.randint(0, 10, (4,6))
X

np.sort(X, axis=0)  #columnwise
np.sort(X, axis=1)  #rowwise
np.sort(X)  #default is rowwise
np.sort(X, axis=None)  #make it 1D and sort

# 88 Partial Sort : Partitioning
x = np.array([7,2,3,1,6,5,4])
np.partition(x,3)  # 3 in betw left less, right more
np.partition(x,5) 

rand = np.random.RandomState(42)
X = rand.randint(0,10, (4,6))
X
np.partition(X, 2, axis=1)  # col
np.partition(X, 3, axis=0) # row
np.partition(X, 3, axis=None)  # 1D

#%% K nearest Neighbours

#Later


#%% Array Concatenation and Splitting

#Concatenation of arrays
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

z = [99, 99, 99]
print(np.concatenate([x, y, z]))

grid = np.array([[1, 2, 3], [4, 5, 6]])

# concatenate along the first axis
np.concatenate([grid, grid])
# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7], [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])
# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])
#Similary, np.dstack will stack arrays along the third axis.

#%%Splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
grid
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)
#https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html#Subarrays-as-no-copy-views

#%% ---------------####


#%%  Pg 92 Structured Data 
name = ['Achal','Apoorva','Goldie','Hitesh']
age = [24, 25, 40, 26]
weight = [67, 72, 73, 77]

#structure array
x = np.zeros(4,dtype=int)
x
data = np.zeros(4, dtype={'names':('name', 'age', 'weight') , 
        'formats':('U10', 'i4', 'f8')})
data  #empty container with variable names
data.dtype
data['name'] = name
data
data['age'] = age
data['weight'] = weight
data  
type(data)   #np array
data['name']
data[0] #1st row
data[data['age'] < 27 ]
data[data['age'] < 27 ]['name']

#above method is dictionary for structure arrays
#next is 
dtype2= np.dtype({'names':('name','age','weight'), 'formats':('U10', 'i4', 'f8') })
data2 = np.zeros(4, dtype=type2)
data2['name'] = name
data2
dtype3= np.dtype([('name','U10'),('age','i4'),('weight','f8')])
data3 = np.zeros(0, dtype=dtype3)

