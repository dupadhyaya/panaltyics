# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:10:53 2017 by Dhiraj Upadhyaya
"""
#pandas 10 Pg 208
import numpy as np
rng = np.random.RandomState(42)
x = rng.rand(10)
x
x = rng.rand(1E6//4)
x




#pandas.eval() 
import pandas as pd
nrows, ncols = 100000, 100
nrows, ncols
rng = np.random.RandomState(42)
rng
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))
df1.head()
%timeit df1 + df2 + df3 + df4
%timeit pd.eval('df1+df2+df3+df4')

np.allclose(df1+df2+df3+df4, pd.eval('df1+df2+df3+df4'))

#Arithmetic Operators

#Comparison

#Bitwise

#Object Attributes


#Columnwise Operations
