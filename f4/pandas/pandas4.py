# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:24:05 2017 by Dhiraj Upadhyaya
"""
#Pandas4
import pandas as pd
import numpy as np
#run this in Ipyton Console
def make_df(cols, ind):
    data = {c:[str(c)+str(i) for i in ind]
        for c in cols}
    return pd.DataFrame(data, ind)

make_df('ABC',range(3))

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df

#Reca;;
x=[1,2,3]
y=[4,5,6]
z=[7,8,9]
np.concatenate([x,y,z])
np.concatenate([x,y,z],axis=1)  #error
x=[[1,2],[3,4]]
np.concatenate([x,x], axis=0)
np.concatenate([x,x], axis=1)
np.concatenate([x,x], axis=2)  #error

#Simple Concatenation with pd.concat
ser1 = pd.Series(['A','B','C'],index=[1,2,3])
ser1
ser2 = pd.Series(['D','E','F'],index=[4,5,6])
ser2
pd.concat([ser1, ser2])

d1 = {'A': ['A1', 'A2'], 'B': ['B1', 'B2']}
df1 = pd.DataFrame(data=d1)
d2 = {'A': ['A3', 'A4'], 'B': ['B3', 'B4']}
df2 = pd.DataFrame(data=d2)
df1
df2
d3 = {'C': ['C3', 'C4'], 'D': ['D3', 'D4']}
df3 = pd.DataFrame(data=d3)
df3
pd.concat([df1,df2])
pd.concat([df1,df3])
pd.concat([df1,df3], axis=0)
pd.concat([df1,df3], axis=1)

#Duplicate indices
x = make_df('AB', [0,1])
y = make_df('AB', [2,3])
y1=y.copy()
x
y
pd.concat([x,y])
y.index = x.index
x.index
y.index
pd.concat([x,y])

#Catching Repeats
pd.concat([x,y], verify_integrity=True)  #duplicate check
pd.concat([x,y], verify_integrity=False)  #duplicate check

y1
pd.concat([x,y1], verify_integrity=True) 

#Ignoring the Index
pd.concat([x,y])
pd.concat([x,y], ignore_index=True)

#Adding Multi-index keys
pd.concat([x,y], keys=['x','y'])

#Concatenation with joins
df5 = make_df('ABC', [1,2])
df6 = make_df('BCD', [3,4])
df5
df6
pd.concat([df5, df6])
pd.concat([df5,df6], join='inner')
pd.concat([df5,df6], join_axes=[df5.columns])
pd.concat([df5,df6], join_axes=[df6.columns])

#Append
df1
df2
df1.append(df2)


#Combining DataSets : Merge and Join

#Categories of Joins
#1 to 1
df1 = pd.DataFrame({'employee':['Achal', 'Apoorva', 'Goldie', 'Hitesh'], 'group':['Engineering','Economics','Engineering','Sales']})
df1
df2 = pd.DataFrame({'employee':['Achal', 'Hitesh', 'Apoorva', 'Goldie'], 'hire_date':[1992,1990, 1991,1980]})
df2
df3 = pd.merge(df1,df2)
df3
df3b = pd.merge(df2,df1)
df3b

#Many to One Joins
df4 = pd.DataFrame({'group':['Engineering', 'Economics', 'Sales'], 'supervisor':['Dhiraj','Prateek','Eeshani']})
df4 
pd.merge(df3,df4)

#Many to Many Joins

df5 = pd.DataFrame({'group':['Engineering', 'Engineering','Economics','Economics', 'Sales','Sales'], 'skills':['python','rpgm','stats', 'finance','managment','excel']})
df5
pd.merge(df1,df5)

#Merge Key
pd.merge(df1, df2, on='employee')
df1
df2
df2d = pd.DataFrame({'name':['Achal', 'Hitesh', 'Apoorva', 'Goldie'], 'hire_date':[1992,1990, 1991,1980]})
pd.merge(df1, df2d, left_on='employee', right_on='name')
pd.merge(df1, df2d, left_on='employee', right_on='name').drop('name',axis=1)
pd.merge(df1, df2d, left_on='employee', right_on='name').drop('employee',axis=1)

#left and right_index
df1a = df1.set_index('employee') 
df2a = df2.set_index('employee')
df1a
df2a
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)

pd.merge(df1a, df2d, left_index=True, right_on='name')


#Speciying Set Arithmetic for Joins
df6 = pd.DataFrame({'name':['Achal', 'Hitesh', 'Apoorva', 'Goldie'], 'python':[76,65,73,80]}, columns=['name','python'])
df6
df7 = pd.DataFrame({'name':['Achal', 'Hitesh'], 'rpg':[85,77]}, columns=['name','rpg'])
df7
pd.merge(df6,df7)
pd.merge(df6,df7, how='inner')
pd.merge(df6,df7, how='outer')
pd.merge(df6,df7, how='left')

df8 = pd.DataFrame({'name':['Achal', 'Hitesh', 'Apoorva', 'Goldie'], 'rank':[1,2,3,4]})
df8
df9 = pd.DataFrame({'name':['Achal', 'Hitesh', 'Apoorva', 'Goldie'], 'rank':[4,2,1,3]})
df9
pd.merge(df8,df9, on='name')
pd.merge(df8,df9, on='name', suffixes=['_L', '_R'])
