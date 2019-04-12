# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:06:55 2017 by Dhiraj Upadhyaya
Chp-3 Numbers and Decimal Numbers
"""

"""
print(7/3)
print(8/5, 8//5)
print(18%7, 2%2, 3%2)


# random Numbers
from random import randint
x = randint(1, 10)
print(' Random no between 1 and 10 is ', x)
# different every time

# Math module
#print(pi)
from math import pi, sin
print(pi, sin(45))

print(abs(-4.3))   # built in functions, no import reqd
print(round(4.3455,2))  # 4.35
print(round(342.3455, -1))  # 340



# help  - no documentation built into python, this is the way to get help
import math  
print(dir(math)) # everything in math module
# ignore starting with _
help(math.floor)
help(math)   # everything in math module

"""

# import everything
from math import *
s= 0
for n in range(1,10001):
    s = s+1 / n**2
print(s)
print(factorial(10))

import numpy

x = numpy.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
y1 = numpy.linspace(0.1, 1.1, 6)
y2 = numpy.arange(0.1, 1.1, 0.2)
y3 = numpy.arange(0.1, 1.11, 0.2)
print(x)

print(x == y1)
# [ True False  True False  True  True]

print(y2)
# [ 0.1  0.3  0.5  0.7  0.9]

print(x == y3)
# [ True False False False False False]

print(y1 == y3)
# [ True  True False  True False False]

print(list(reversed(numpy.linspace(1.1, 0.1, 6))) == y1)
# [ True  True  True  True False  True]
#x1 = {1,2,3,4,5}
#print(reversed(x1))
x3 = numpy.arange(1, 10, 2)
print(x3)