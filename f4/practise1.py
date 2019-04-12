# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:55:14 2018
Dhiraj
@author: du
"""

name = 'dhirajupadhyaya dhiraj'
key ='abcdefghijklmneopqrstuvwxyz'

msg = input('Input your message')
msg.lower()
lmsg = msg.lower()

'a'.isalpha()

for c in lmsg:
    print(c)
    if c.isalpha():
        print(key[name.index(c)], end='')

L=[1,2,3]
L
L1= input('Enter a no')
L1
L2=eval(input('Enter a no'))
L2
L3=eval(input('Enter a nos with space'))
L3

len(L)

for i in L:
    print(i)

s = input()
numbers = list(map(int, s.split()))
numbers
L=[]
for i in range(10):
    L.append(i)
L

import random
print(random.randrange(1,2,10))
random.randrange(2,3,10)
random.seed(0)
random.random()
random.randrange(1, 10, 4)
