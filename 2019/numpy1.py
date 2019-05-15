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
