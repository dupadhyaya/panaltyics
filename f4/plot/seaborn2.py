# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:49:00 2017 by Dhiraj Upadhyaya
"""

#seaborn 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set
rng = np.random.RandomState(0)
x = np.linspace(0,10, 500)
y = np.cumsum(rng.randn(500,6),0)
print(y.sum())
plt.plot(x,y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()
