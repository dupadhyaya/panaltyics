# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 07:41:48 2017 by Dhiraj Upadhyaya
"""
import pandas as pd
import numpy as np

#Stack
tuples = list(zip(*[['ds', 'ds', 'ds', 'mls','mls'],['graduate', 'diploma', 'pgdegree','graduate', 'pgdegree']]))
tuples
index = pd.MultiIndex.from_tuples(tuples, names=['dept', 'course'])
index
np.random.randint(10, size=10)
np.random.randint(10)

np.random.randint(10, size=(5, 2))

data=[[3,1],[4,2],[10,3],[5,2],[8,5]]
data=[[3,4,10,5,8],[1,2,3,2,5]]
data
df = pd.DataFrame(data, index=index, columns=['students', 'faculties'])
df

df = pd.DataFrame(np.random.randint(10, size=(5,2)), index=index, columns=['students', 'faculties'])

df
