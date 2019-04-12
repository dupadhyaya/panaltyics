# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 06:13:11 2018
Dhiraj
@author: du
"""

#Shallow and Deep Copy
# Python code to demonstrate copy operations
 
# importing "copy" for copy operations
import copy
 
# initializing list 1
li1 = [1, 2, [3,5], 4]
li1 
# using deepcopy to deep copy 
li2 = copy.deepcopy(li1)
li2 
# original elements of list
print ("The original elements before deep copying")
for i in range(0,len(li1)):
    print (li1[i],end=" ")
 
print("\r")
 
# adding and element to new list
li2[2][0] = 7
li2 
li1
# Change is reflected in l2 
print ("The new list of elements after deep copying ")
for i in range(0,len( li1)):
    print (li2[i],end=" ")
 
print("\r")
 
# Change is NOT reflected in original list
# as it is a deep copy
print ("The original elements after deep copying")
for i in range(0,len( li1)):
    print (li1[i],end=" ")


# Shallow Copy
# Python code to demonstrate copy operations
 
# importing "copy" for copy operations
import copy
 
# initializing list 1
li1 = [1, 2, [3,5], 4]
 
# using copy to shallow copy 
li2 = copy.copy(li1)
li2

# original elements of list
print ("The original elements before shallow copying")
for i in range(0,len(li1)):
    print (li1[i],end=" ")
 
print("\r")
 
# adding and element to new list
li2[2][0] = 7
li2
li1
# checking if change is reflected
print ("The original elements after shallow copying")
for i in range(0,len( li1)):
    print (li1[i],end=" ")
    

#3rd Method of Copy: Function of Object
li1 = [1, 2, [3,5], 4]
li1
li2 = li1.copy()
li2
li2[3]=99
li2
li1
#copy pointing to other object/memory
id(li1)
id(li2)
id(li3)



