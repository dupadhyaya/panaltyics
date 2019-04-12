# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:25:34 2017 by Dhiraj Upadhyaya
"""
#Script myplot.py
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
