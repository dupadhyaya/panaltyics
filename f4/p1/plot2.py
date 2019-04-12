# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:22:17 2017 by Dhiraj Upadhyaya
Plot2
"""
"""
# importing the required module
import matplotlib.pyplot as plt
 
# x axis values
x = [1,2,3]
# corresponding y axis values
y = [2,4,1]
 
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
 
# giving a title to my graph
plt.title('My first graph!')
 
# function to show the plot
plt.show()
"""

import matplotlib.pyplot as plt
 
# line 1 points
x1 = [1,2,3]
y1 = [2,4,1]
# plotting the line 1 points 
plt.plot(x1, y1, label = "line 1")
 
# line 2 points
x2 = [1,2,3]
y2 = [4,1,3]
# plotting the line 2 points 
plt.plot(x2, y2, label = "line 2")
 
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Two lines on same graph!')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()

plt.new()
# ----
#%matplotlib notebook
import numpy as np

import matplotlib.pyplot as plt
x = np.linspace(0, 3*np.pi, 500)
plt.plot(x, np.sin(x**2))
plt.title('A simple chirp')
plt.show()
#----
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(range(100))

%matplotlib qt5
import matplotlib.pyplot as plt
plt.plot(range(100))
plt.show()
