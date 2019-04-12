# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:59:07 2018 by Dhiraj Upadhyaya
"""

#list 
x = [1,2,3]
x
print(x)

rollnos = [1,2,3,4,5,6,10,11]
rollnos[0]
rollnos[8]
rollnos[7]
len(rollnos)
sum(rollnos)
min(rollnos)
max(rollnos)


rollnos
#Run Together next 2 lines
for i in range(len(rollnos)):
    print(rollnos[i], end=',')
    
if 13 in rollnos:
    print('Yes')
else:
    print('No')
rollnos

rollnos.append(13)
rollnos
sorted(rollnos)
rollnos.index(13)

#Reverse a list
type(rollnos)
rollnos.reverse()
rollnos # check the list now
#2nd method
rollnos[::-1]
rollnos

#3rd method : orginal not changed
rollnos
for item in reversed(rollnos):
    print(item, end=', ')

#3a Method
rollnos
list(reversed(rollnos))

#Sort
rollnos.sort()
rollnos
import random
random.shuffle(rollnos)
rollnos
rollnos.sort()
rollnos

rollnos.remove(1)
rollnos
del rollnos[1]
rollnos
del rollnos[:2]
rollnos

random.choice(rollnos)
random.sample(rollnos,2)

' '.join(str(rollnos))

# Create List
L2 = [i for i in range(3)]
L2
L2 = [i for i in range(3,10)]
L2
L2=[]
for i in range(2):
    for j in range(2):
        L2.append([i,j])
L2
L={(i,j) for i in range(2) for j in range(4)}
L
L=[(i,j) for i in range(2) for j in range(4)]
L
type(L)
L3 = [j for i in range(2, 8) for j in range(i*2, 50, i)]
L3
type(L3)
L2= [i for i range(2) for j in range(3)]
L2
