# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:19:30 2017 by Dhiraj Upadhyaya
Data Camp Functions & Numpy
"""
"""
fam = [1.73, 1.68, 1.71, 1.81]
print(fam)
max(fam)
print(round(1.68,1))
myRoundedList = [ round(elem, 1) for elem in fam ]
print(myRoundedList)
"""
"""
fam2 = ['liz',1.73,'emma', 1.68, 'mom', 1.71, 'dad', 1.81, 1.71 ] # list of mixed values int char
print(fam2)
print(' Index of emma ', fam2.index('emma'))  # serial position
print(' Index of 1.71 ', fam2.index(1.71))  # serial position

print(' Count of 1.71 ', fam2.count(1.71))  # how many times a value has occured
print(' Count of emma ',fam2.count('emma'))
#print(' Index of emma ', fam2.value('emma'))  
"""
fam3 = ['liz','emma',  'mom', 'dad' ]
#print('liz'.capitalise())
#print(elem.capitalise() for elem in fam3)
FAM3 = [x.upper() for x in ["a","b","c"]]
print(FAM3)
FAM3 = [x.upper() for x in fam3]
print(FAM3)
# map(lambda x:x.upper(),["a","b","c"])

sister = 'liz'
height = 1.73
print(sister, height, sep=' ')
print(sister.capitalize())  # initial capital
print(sister.upper())  # all upper case 
print(sister.replace('z', 'SA'))

print(fam3)
#fam3.replace('mom','mummy')  # cannot replace in list
print(sister.index('z'))  # position of character in a string
fam3.append('dhiraj')  # method does not return
print(fam3)
print(type(fam3))  # list
