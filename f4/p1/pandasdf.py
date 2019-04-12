# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:40:45 2017 by Dhiraj Upadhyaya
Pandas
"""
"""
import pandas
import numpy as np
df = DataFrame(np.random.rand(4,4), columns = list('abcd'))
print(df)

print('print1',df.loc[:, df.columns != 'b'], sep='\n')
print('print2', df.drop('b', axis=1),sep='\n')
print('print3', df.drop(['a', 'b'], axis=1) ,sep='\n')
print('print4', df[df.columns.difference(['b'])] ,sep='\n')
print('print5', df[[i for i in list(df.columns) if i != 'a']], sep='\n')
print('print6', df.loc[0:2] ,sep='\n')
df1 = df.loc[:,'a':'c']
print('print7', df1 ,sep='\n')
df2 = df.loc[:, ['a', 'c']]
print('print8', df2 ,sep='\n')
df3 = df.loc[:, 'a':'b']
print('print9',df3 ,sep='\n')
df4 = df[['a','b']]
print('print10', df4 ,sep='\n')
#df5 = df.ix[:,0:2] # ixdepreciate
df5 = df.iloc[:,0:3] # 
print('print11', df5 ,sep='\n')
#(df.columns) are ['index','a','b','c']
df6 = df[df.columns[2:4]] # Remember, Python is 0-offset! The "3rd" entry is at slot 2.
print('print12', df6, sep='\n')
df7 = df[df.columns[3:4]] # Remember, Python is 0-offset! The "3rd" entry is at slot 2.
print('print13', df7, sep='\n')

#df.ix slices columns a bit more concisely, but the .columns slicing inte
#.ix is deprecated. Please use
#.loc for label based indexing or
#.iloc for positional indexing

"""
import pandas as pd
import numpy as np
np.random.seed(5)
"""
df = pd.DataFrame(np.random.randint(100, size=(100, 6)), 
                  columns=list('ABCDEF'), 
                  index=['R{}'.format(i) for i in range(100)])
print(df.head())
print(df.loc['R3':'R5', ])

#print(df.loc[:, 'C':'E'])
print(df.loc['R6':'R10', 'C':'E'])
print(df.loc['R2':'R5', df.columns.isin(list('ABD'))])
#print(df.loc[df.items.isin('R1'), df.columns.isin(list('ABD'))])
print(df.iloc[2], sep=' ', end='\n')
print(df[2:3])

#columns = ['b', 'c']
#df1 = pd.DataFrame(df, columns=columns)
#colsToDrop = ['a']
#df.drop(colsToDrop, axis=1)
#iloc[row slicing, column slicing]
"""

df = pd.DataFrame( np.random.randint(100, size=(100, 6)))
#print(df)
#print(df.loc[2:3,0:1])
#print(df.iloc[[2]])
#print(df.loc[[2]])
#print(df._slice(slice(0, 2)))
#print(df._slice(slice(0, 2), 0))
#print(df._slice(slice(0, 2), 1))
print(df.values[99])
print(df.size)

print(df.shape)
print(df.shape[1])
print(df.shape[0])
#print(df.count + 1)
#print(df[1].count + 1)



for i in range(0,df.shape[0]):
    print('Row - ', i, df.values[i], sep=' ')

def nrow(df):
    print("{:,}".format(df.shape[0]))
nrow(df)