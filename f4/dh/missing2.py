# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:26:24 2017 by Dhiraj Upadhyaya
"""
import pandas as pd
import numpy as np

vals1 = np.array([1, None, 3, 4])
vals1
sum(vals1)


#Missing Values
import pandas as pd
import numpy as np
vals2 = np.array([1,np.nan, 3,4])
vals2
1 + np.nan
vals2.sum()
vals2.min()
vals2.max()
np.nansum(vals2)
np.nanmax(vals2)

pd.Series([1, np.nan, 2, None])
#upcast to floating point due None
#x = pd.Series(range(2), dtype=int)
#x
#x[0] = np.nan
#x



#Operating on Null Values
#is.null(), notnull, dropna, fillna

#detecting Null Values
data = pd.Series([1, np.nan, 'hello', None])
data
data.isnull()
data[data.notnull()]

data.dropna()

# for df, it will drop na values by default
df.dropna()
df
df.dropna(axis='columns')

df.drop


np.random.seed(0)
data = np.array(np.random.randint(1,100,50))
data

list('ABCDE')
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
