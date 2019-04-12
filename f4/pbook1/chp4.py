# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:05:08 2017 by Dhiraj Upadhyaya
Chp-4 : If statements
"""
from random import randint
"""
num = randint(1,10)
guess = eval(input('Enter your guess '))
if guess == num :
    print('Your guess was correct ')
else:
    print('Your guess was not correct, no was ', num)

"""
# Conditional Operators
grade = randint(80,95)
if grade >= 80 and grade < 90:
    print(' (and) Your grade is B ', grade)
# next time another random no
if grade >= 80 & grade < 90:
    print(' (&) Your grade is B ', grade)

if grade >= 80 or grade < 90:
    print(' (OR) Your grade is C ', grade)

if grade >= 80 | grade < 90:
    print(' (|) Your grade is C ', grade)

score=10005; time=30
if not( score>1000 or time > 20):
    print('Game continues')