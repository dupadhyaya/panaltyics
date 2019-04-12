# -*- coding: utf-8 -*-
"""
Fri Feb  9 14:08:48 2018: Dhiraj
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
