# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:21:23 2017 by Dhiraj Upadhyaya
"""

#arrays3
import numpy as np
import math

np.random.seed(0)
data = np.random.uniform(5,10,2*3*4)
len(data)
data= np.ceil(data)
len(data)
data2 = data.view('int32')
len(data2)
