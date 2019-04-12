# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:24:16 2018
Dhiraj
@author: du
"""
import random

L1 = [3, 6, 8, 2, 78, 1, 23, 45, 9]
sorted(L1)
L1
random.shuffle(L1)
L1.sort()
L1


random.shuffle(L1)
sorted(L1, reverse=True)

random.shuffle(L1)
L1.sort(reverse=True)





#Tuple
T1 = (3, 6, 8, 2, 78, 1, 23, 45, 9)
sorted(T1)
random.shuffle(T1) # values cannot be changed

def getKey(item):
    return item[0]

L2 = [[2, 3], [6, 7], [3, 34], [24, 64], [1, 43]]
sorted(L2, key=getKey)

def getKey(item):
    return item[1]
L2 = [[2, 3], [6, 7], [3, 34], [24, 64], [1, 43]]
sorted(L2, key=getKey)



#Dictionaries
D1 = {'first': 1, 'second': 2, 'third': 3, 'Fourth': 4}
list(D1)
sorted(D1)
sorted(D1.values())
sorted(D1.keys())
