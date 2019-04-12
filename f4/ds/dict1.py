# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:41:08 2018
Dhiraj
@author: du
Dictionary
"""
D1={}
D2={1:'a',2:'b',3:10}
D2


# Create from a list
L1 = ['achal','apoorva','hitesh']
D1 = dict(zip(range(len(L1)),L1))
D1

D1={1:'a', 2:'b'}

# String Generator
s='a=apple,b=boy,c=cat,d=dog,e=elephant'
D1=dict(i.split("=") for i in s.split(","))
D1

#Start from empty Dict
D2={}
D2[3]='c'
D2
D2.update({4:'d'})
D2

D2.update({5:'e', 6:'f'})
D2


K1=[8,9,10]
V1=['g','h','i']
for k,v in zip(K1,V1):
    D2[k] =v
D2

D = {**D1, **D2}
D

D = dict(list(D1.items()) + list(D2.items()))
D
#1 to many from lists Associating Multiple Values with Each Key in a Dictionary

import collections

code = [1,1,1,1,2]
subject = ["Maths","Science","Maths","Physics","English"]

d = collections.defaultdict(list)

for k,v in zip(code, subject):
    d[k].append(v)
d
type(d)
d[1]


my_dict = {}
for k, v in zip(code, subject):
    my_dict.setdefault(k, []).append(v)

my_dict


#List Comprehensions

#squares
squares = {x: x+1 for x in range(6)}

# Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
print(squares)

#odd squares
odd_squares = {x: x*x for x in range(11) if x%2 == 1}
odd_squares
# Output: {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}
print(odd_squares)

d1 = {n: n*n for n in range(7) }
d1
d2= { val:key for key, val in d1.items()}
d2

l3 = [11,12,13,15,12,19]
d3 = { key:l3.count(key) for key in set(l3)}
d3
