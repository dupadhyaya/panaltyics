# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:52:25 2018
Dhiraj
@author: du
"""
#Sample List and their types
L1=[1,2,3,4,5]

L2=[1,2,3,'a',True]
L3=[i for i in range(5)]
L1
L2
L3
type(L1)
type(L2)
type(L3)
type(L3[4])
for i in range(len(L2)):
    print(type(L2[i]), end=' ')
#Functions
L=L1 + L2
L
sum(L)  # error
sum(L[1:4])   # do numeric functions on numeric values
L[1:4]
L[len(L)-2].upper()  # do all character functions on this index value

# Lists inside a List
L4 = [1,2,L2]
print(L4)
L4[1]
L4[2]  # L4[2][0]
L4
L4[2][2]  # understand this

# Multiple Levels
L5a=[1,2,3,4,5]
L5b=[6,7,8,9,10]
L5=[L5a,L5b]
L5
L5[1]
L5[0][3]
L5[1][4]

# Other Data Types
D1 = {1:'achal', 2:'apoorva',3:'hitesh','dean':'dhiraj'}
D1
D1.keys()
D1.values()
D1.items()
D1[1]
D1['dean']

D2 = {'a':L1,'b':L3}
D2
D3 = {'a':L1,'b':L2}
D3
#D3.pop('b')
D3['c']=D3.pop('b')   # change name of key
D3
D3['c'][3]='z'  # a to z
D3
vars(D1)
for key in D2:
	print(key, end=' ')
	print(D2[key], end=' ')




#List to a Set
L1=[1,2,3,4,5,5]
S1 =set(L1)
type(S1)
S1
