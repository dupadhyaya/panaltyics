# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:26:24 2017 by Dhiraj Upadhyaya
"""

#Missing Values
import pandas as pd
import numpy as np
vals = np.array([1,np.nan, 3,4])
vals

np.random.seed(0)
data = np.array(np.random.randint(1,100,50))
data

df = pd.DataFrame(np.reshape(data, (10,5)))
df.columns=[list('ABCDE')]
df

df[df < 30] = np.nan
df
df.sum(), df.min(), df.max()
count_nan = len(df) - df.count()
count_nan
df.shape[1] - df.count(axis=1)


df.isnull().sum(axis=1)
df.isnull().sum(axis=0)
df.isnull().sum(axis=0).tolist()

df.isnull().sum(axis=1).tolist()

df.isnull().any().tolist()
df.isnull().sum().sum() #total in df
df.isnull().sum().sum()
df.isnull().sum(axis=0).tolist()

df.dropna(axis='columns',how='any',inplace=False)
df
df.isnull().sum(axis=1).tolist()
df.dropna(axis='rows',thresh=3, inplace=False)
df

#Filling NA values
df.fillna(9999,inplace=False)
df
df.fillna(method='ffill')
df.fillna(method='bfill')
df
df.fillna(method='ffill',axis=1)#colfwd
df.fillna(method='bfill',axis=1)
