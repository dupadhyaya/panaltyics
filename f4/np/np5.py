# -*- coding: utf-8 -*-
"""
Numpy : Shape and axis
"""
import numpy as np
np.arange(100)
np.arange(100).reshape(100,1) # rows-100, col-1
np.arange(100).reshape(1,100)
np.arange(100).reshape(20,5)  #rows-20, col-5
np.arange(100).reshape(10,2,5)
sum(np.arange(100))  # sum
sum(np.arange(100))/len(np.arange(100))  # avg
np.arange(100).mean()
np.arange(100).reshape(100,1).mean()
np.arange(100).reshape(20,5).mean()
np.arange(100).reshape(20,5).mean(axis=1)  # 20 rows
np.arange(100).reshape(20,5).mean(axis=0) # 5 cols
np.arange(100).reshape(20,5).mean(axis=2) # error

np.arange(100).reshape(10,2,5).mean(axis=2)
np.mean(np.arange(100).reshape(10,2,5))
np.mean(np.arange(100).reshape(10,2,5),axis=2)
np.mean(np.arange(100).reshape(10,2,5),axis=0)
