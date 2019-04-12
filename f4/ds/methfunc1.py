# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:23:07 2018 by Dhiraj Upadhyaya for DS 
Functions and Methods
"""

#A function is a piece of code that is called by name. It can be passed data to operate on (i.e., the parameters) and can optionally return data (the return value). All data that is passed to a function is explicitly passed

def sum2(num1, num2):
    return num1 + num2

sum2(1,2)

"""
A method is a piece of code that is called by name that is associated with an object. In most respects it is identical to a function except for two key differences.

It is implicitly passed for the object for which it was called.
It is able to operate on data that is contained within the class (remembering that an object is an instance of a class - the class is the definition, the object is an instance of that data). """

class Human:
    def my_method(self):
        print("I am a Human")
    x=[1,2,3]
    y=('a','b','c')
h1 = Human()
h1.my_method()  # Prints "I am a Human"
h1.x
h1.y
h1.x.count(1)
h1.x.reverse()
h1.x
