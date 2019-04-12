# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:24:08 2017 by Dhiraj Upadhyaya
assignment by RDG
"""

"""
# program to add 2 nos entered by users

num1 = eval(input('First No ') )
num2 = eval(input('2nd No ' ))
print('Sum of 2 nos - ', num1, '  and ', num2, ' is ', num1+num2)

"""

# check if no is positive or negative or zero
"""
num = eval(input('Enter a number '))
if num == 0:
    print(' The number ', num , 'is 0 ')
if num > 0:
    print(' The number ', num , 'is positive ')
if num < 0:
    print(' The number ', num , 'is negative ')

"""

# A3  : return absolute diff betw n and 21

def diff21(n, x=21):
    print(' The difference between ', x , ' and ', n , ' is ' ,  abs(x - n))

diff21(19)
diff21(10)
diff21(0)
diff21(31)


#A4 : sum_double, 2 int values - sum, if same, double the same

def sum_double(x, y):
    if x == y:
        return((x + y) * 2)
    else:
        return(x + y)
#m=1;n=2
m=2;n=2

result= sum_double(m,n)
print(' The sum of two nos - ', m, ' and ', n, ' is ', result)

#x = int(input("Enter a number: "))
#print(x, type(x))

# Missing Char

def missing_char(string, n):
        return ''.join(x for x in string if string.index(x) != n)

print(missing_char('kitten', 1))
print(missing_char('kitten', 0))
print(missing_char('kitten', 4))

#arr = map(int, input().split())   
#print(arr)
#num1, num2 = map(int, input().split())