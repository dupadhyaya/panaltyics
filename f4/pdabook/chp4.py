# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:25:34 2017 by Dhiraj Upadhyaya
"""
#Importing matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt  # this is used most often

#aesthetics style like aes in ggplot
plt.style.use('classic')
# Plotting from Script 
#run together or create a file and run it python myplot.py
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()


#%% Plotting from IPython shell
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
#need to force update by 
plt.draw()
#plt.show() is not required


#%% IPython notebook
%matplotlib notebook  # interactive plots 
%matplotlib inline # static images in notebook

#this book has used %matplotlib inline
#run once only per kernel session
%matplotlib inline
import numpy as np
x = np.linspace(0,10,100)
x
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

# saving Figures
fig.savefig('myfigure3.png')

#confirm
!dir *.png

from IPython.display import Image
# Image('my_figure.png')


#%% Two Interfaces
#Matlab style
%matplotlib inline
plt.subplot(2,1,1)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))

#Object Oriented Interface
fig, ax = plt.subplots(3)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
ax[2].plot(x, np.tan(x))


