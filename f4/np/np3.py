# -*- coding: utf-8 -*-
"""
NP Creation
"""
#Converting Python array_like Objects to NumPy Arrays
import numpy as np
x = np.array([2,3,1,0])
x
type(x)
x = np.array([2, 3, 1, 0])
x
x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists, and types
x
x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])
x

#Intrinsic NumPy Array Creation
np.zeros((2, 3)) 

np.arange(10)

np.arange(2, 10, dtype=np.float)

np.arange(2, 3, 0.1)


np.linspace(1., 4., 6)

np.indices((1,2))
np.indices((0,0))    

np.indices((1,1))    
np.indices((2,2))    
np.indices((3,3))


#=-----
a[(0,1,2,3,4),(1,2,3,4,5)]
?a
