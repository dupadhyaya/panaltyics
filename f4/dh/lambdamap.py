# -*- coding: utf-8 -*-
"""
Mon Feb 19 15:34:01 2018: Dhiraj
"""
#lambda Functions

f = lambda x, y : x + y
f(1,1)

add = lambda x, y : x + y 
print(add(2, 3)) # Output: 5

def fahrenheit(T):
    return ((float(9)/5)*T + 32)
def celsius(T):
    return (float(5)/9)*(T-32)
temp = (36.5, 37, 37.5,39)

F = map(fahrenheit, temp)
F
C = map(celsius, F)

Celsius = [39.2, 36.5, 37.3, 37.8]
Fahrenheit = map(lambda x: (float(9)/5)*x + 32, Celsius)
print(Fahrenheit)
C = map(lambda x: (float(5)/9)*(x-32), Fahrenheit)
print(C)



fib = [0,1,1,2,3,5,8,13,21,34,55]
result = filter(lambda x: x % 2, fib)
print(result)
result = filter(lambda x: x % 2 == 0, fib)
result

reduce(lambda x,y: x+y, [47,11,42,13])

f = lambda a,b: a if (a > b) else b
reduce(f, [47,11,42,102,13])

from functools import reduce
reduce( (lambda x, y: x * y), [1, 2] )
reduce( (lambda x, y: x * y), [1, 2,3] )
reduce( (lambda x, y: x * y), [1, 2, 3, 4] )
reduce( (lambda x, y: x / y), [1, 2, 3, 4] )

#loop
L = [1, 2, 3, 4]
result = L[0]
for x in L[1:]:
	result = result * x
result

#
import math
def square_root(x): return math.sqrt(x)
square_root = lambda x: math.sqrt(x)
list(square_root)
#filter and lambda
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(filter(lambda x: (x%2 == 0) , my_list))
# Output: [4, 6, 8, 12]
print(new_list)

a = [1, 2, 3, 4, 5, 6]
list(filter(lambda x : x % 2 == 0, a))
