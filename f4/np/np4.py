# -*- coding: utf-8 -*-
"""
NP
"""
import numpy as np
#1-D:
a = np.array([0, 1, 2, 3])
a
a.ndim
a.shape
len(a)

#2D
b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
b
b.ndim
b.shape
len(b)     # returns the size of the first dimension


c = np.array([[[1], [2]], [[3], [4]]])
c
c.shape
c.ndim

#----Functions for creating arrays
# In practice, we rarely enter items one by one…
#Evenly spaced:

a = np.arange(10) # 0 .. n-1  (!)
a

b = np.arange(1, 9, 2) # start, end (exclusive), step
b

#or by number of points:

c = np.linspace(0, 1, 6)   # start, end, num-points
c

d = np.linspace(0, 1, 5, endpoint=False)
d

#Common arrays:

a = np.ones((3, 3))  # reminder: (3, 3) is a tuple
a

b = np.zeros((2, 2))
b

c = np.eye(3)
c
d = np.diag(np.array([1, 2, 3, 4]))
d

#np.random: random numbers (Mersenne Twister PRNG):

a = np.random.rand(4)       # uniform in [0, 1]
a  

b = np.random.randn(4)      # Gaussian
b  

np.random.seed(1234)        # Setting the random seed
np.random.randn()
np.random.seed(1234)        # Setting the random seed
np.random.randn()



#Data Types
a = np.array([1, 2, 3])
a.dtype

b = np.array([1., 2., 3.])
b.dtype

c = np.array([1, 2, 3], dtype=float)
c.dtype

a = np.ones((3, 3))
a.dtype


d = np.array([1+2j, 3+4j, 5+6*1j])
d.dtype

e = np.array([True, False, False, True])
e.dtype

f = np.array(['Bonjour', 'Hello', 'Hallo',])
f.dtype     # <--- strings containing max. 7 letters  

#Much more:	int32 int64 uint32 uint64
a = np.arange(10)
a

a[0], a[2], a[-1]

a[::-1]
a = np.diag(np.arange(3))
a
a[1, 1]

a[2, 1] = 10 # third line, second column
a



a[1]




#NP operations
import numpy as np
L = np.array([1,1,3,1,4,5,8])
list(L).count(1)  #3
(L == 1).sum()  #3
sum(L == 1) #3
from collections import Counter
Counter(L)  #Counter({1: 3, 8: 1, 3: 1, 4: 1, 5: 1})
(L%2).sum()  #5 how many are odd

