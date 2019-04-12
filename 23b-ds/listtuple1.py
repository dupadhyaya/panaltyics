# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:30:08 2018 by Dhiraj Upadhyaya for DS 
Tuples
"""
L1 = [1,2,'DS','Dhiraj', True]
L2=[i for i in range(9)]
L2
sum(L2[:])
L1[3].upper()
L1.count()
sum(L1[0:2])

if L1[4]:
    print('hello')
else:
    print('No Hello')       

if not L1[4]:
    print('Not True')
else:
    print('Hellow')

L3 = [1,2,3, {101:'S1'}, {1,2,3}]
L3 
L3[1]
L3[3]
L3[3].values()
L3[3].keys()
L3[3].items()
L3[3].items()


type(L3[2])

M = ('dept',(23,'MG road'),[50,34,'India'])

