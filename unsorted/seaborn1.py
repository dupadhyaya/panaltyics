# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:24:57 2017 by Dhiraj Upadhyaya
"""

#Seaborne1
import matplotlib.pyplot as plt
#import matplotlib as plt

plt.style.use('classic')
#%matplotlib inline
import numpy as np
import pandas as pd

#data
rng = np.random.RandomState(0)
x = np.linspace(0,10, 500)
x
y = np.cumsum(rng.randn(500,6),0)
y
print(y.sum())
y.mean()
len(y)
plt.plot(x,y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()

#run through prompt : c>python seaborn1.py
#select lines 24 to 26 and run as selection
