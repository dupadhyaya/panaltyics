# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:47:50 2018
Dhiraj
@author: du
"""
#map

Lst1 = [10, 20, 30, 40]
Lst2 = [50, 60, 70, 80]

sum_Lsts = map(lambda num1,num2: num1+num2, Lst1,Lst2)
print("The sum of two lists", list(sum_Lsts))

list(map(min, [1,2,3,4], [0,10,0,10]))

list(map(lambda x: x+1, [1,2,3]))

list(map(chr, [66, 53, 0, 94]))
[*map(chr, [66, 53, 0, 94])]

def calctwolsts(num1,num2):
   return num1*num2
 
L1 = [2, 4, 6, 8, 10]
L2 = [5, 10, 15, 20]
L3 = map(calctwolsts, L1, L2)
print(list(L3))


def calCelsius(F):
     C = ((5.0/9.0)*(F - 32))
     return C
fah_temp = [98.5, 101,102,203,104]
conv_temp = (map(calCelsius, fah_temp))
#Convert the returned map object to list
Cel_lst = list(conv_temp)
print(Cel_lst)

my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(map(lambda x: x * 2 , my_list))
# Output: [2, 10, 8, 12, 16, 22, 6, 24]
print(new_list)



def square(x):
        return (x**2)
def cube(x):
        return (x**3)

funcs = [square, cube]
for r in range(5):
    value = list(map(lambda x: x(r), funcs))
    print(value)
