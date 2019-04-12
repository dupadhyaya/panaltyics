# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:58:09 2018
Dhiraj
@author: du
"""

#Arrays
import array
x=array.array('l')
x
type(x)
array.array('u', 'hello \u2641')
array.array('l', [1, 2, 3, 4, 5])
array.array('d', [1.0, 2.0, 3.14])


a = array.array("B", range(16)) # unsigned char
a
b = array.array("h", range(16)) # signed short
b

print(repr(a.tostring()))

c = array.array('B',[1,2,3])
c
?array.array
sum(c)
c.reverse()
c


#------------------
import array
a = array.array("B", [1, 2, 3])
a.append(4)
a = a + a
a
a = a[2:-2]
a
repr(a.tostring())
for i in a:
    print(i)

b=array.array('B', [3, 4, 1, 2])
b.count(3)
sum(b)
b.count()
c=b.tolist()
type(c)
b.typecode
b.itemsize
len(b)
sum(b)
array.typecodes


# 
from numpy import *
for i in arange(5):
    print(i)

a = arange(5)
a
a.s
