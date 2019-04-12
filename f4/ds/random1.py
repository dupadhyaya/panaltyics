# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:02:11 2018 by Dhiraj Upadhyaya for DS 
"""

#Random Nos
from random import randint
#in console windows type above till import and then press tab to see choice
x=randint(1,10)
x

import random
#10 nos between 1 and 101
for x in range(10):
    print(random.randint(1,101), end=' ')

#random nos multiple of 5
for x in range(10):
    print(random.randint(1,21)* 5, sep=' ', end=';')

L1=['win', 'lose', 'draw']
random.choice(L1) 

S1 = 'the quick brown fox jumps'
L2 = S1.split()
L2
random.shuffle(L2)
L2

# Samples
# 10 nos in range 0,100
L3=[randint(0,100) for p in range(0,10)]
L3
random.sample(L3, k=4)

data = 1,2,3,4,5
data
type(data)
random.choices(data,k=2)
