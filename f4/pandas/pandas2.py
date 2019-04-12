# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:27:04 2017 by Dhiraj Upadhyaya
"""
#Pandas2
#Operating on Data in Pandas
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0,10,4))
ser

df = pd.DataFrame(rng.randint(0,10,(3,4)), columns=['A','B','C','D'])
df
ser
np.exp(ser)
np.exp(1)
np.sin(df * np.pi / 4)

#Ufuncs : Index Alignment

population = pd.Series({'Delhi':25000, 'Noida':10000, 'Gurgaon':15000,'Faridabad':7500,'Ghaziabad':8000})
population
area  = pd.Series({'Delhi':100, 'Noida':30, 'Gurgaon':40,'Faridabad':25,'Ghaziabad':20})
area
population/area
area.index | population.index

A = pd.Series([2,4,6], index=[0,1,2])
B = pd.Series([1,3,5], index=[1,2,3])
A, B
A+B
A.add(B)
A.add(B, fill_value=0)

#Index Alignment
A = pd.DataFrame(rng.randint(0,20,(2,2)), columns=list('AB'))
A
B = pd.DataFrame(rng.randint(0,20,(3,3)), columns=list('BAC'))
B
A+B
fill = A.stack().mean()
fill
A.add(B, fill_value=fill)  #fill with mean values of A

#Operations between DF and Series

A=rng.randint(10, size=(3,4))
A, A[0]
A - A[0]
df = pd.DataFrame(A, columns=list('QRST'))
df

df.subtract(df['R'],axis=0)
halfway = df.iloc[0, ::2]
halfway
df - halfway


#Handling Missing Data
import numpy as np
import pandas as pd
vals1 = np.array([1, None, 3, 4])
vals1
"""
for dtype in ['object', 'int'] :
    print('dtype =', dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()

dtype=object
"""
vals1.sum()  #error

#NaN : Missing numerical data
vals2 = np.array([1, np.nan, 3,4])
vals2.dtype
vals2
1 + np.nan
0 * np.nan
vals2.sum(), vals2.min(), vals2.max()
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

#NaN and None in Pandas
pd.Series([1, np.nan, 2, None])
x = pd.Series(range(2), dtype=int)
x[0] = None
x

x = pd.Series(range(2),dtype=int)
x
x[0] = None
x
#Tricky Pg 124

#Operating on Null Values
#isnull(), notnull(), dropna(), fillna()
data = pd.Series([1, np.nan, 'Hello', None])
data
data.isnull(), data.notnull()
data.dropna()

df = pd.DataFrame([[1,   np.nan, 2], [2,   3,  5 ],[np.nan, 4,   6 ]])
df
df.dropna()
df.dropna(axis=0)
df.dropna(axis='rows')

df.dropna(axis=1)
df.dropna(axis='columns')
df
df[3] = np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='columns', how='any')
df.dropna(axis='columns', thresh=3)
df.dropna(axis='columns', thresh=2)

#Filling null values Pg 126
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
data.fillna(99)
data.fillna(method='ffill')
data.fillna(method='bfill')

df
df.fillna(99)
df.fillna(method='ffill')
df.fillna(method='ffill', axis='rows')
df.fillna(method='ffill', axis='columns')
