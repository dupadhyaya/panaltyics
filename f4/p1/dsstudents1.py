# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:59:33 2017 by Dhiraj Upadhyaya
"""
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
df = pd.read_csv('./data/dsstudents.csv', index_col=0)
print(df)
df1 = DataFrame(df)
print(df1.loc[:,'name'])
df1.describe(include='all') # summary of DF
df1.describe(include=[np.number])
df1.describe(include=[np.object])
"""
print(df1['gender'][:5])  # first 5 rows showing gender only
gender_counts = df1['gender'].value_counts()
print(gender_counts)  # table of gender
print(gender_counts[:2])
#gender_counts.plot(kind='barh',rot=0)

print(df1.loc[:,'course']) # rowname and course
print(df1.iloc[0:2, 0:5]) #First 2 rows, first 5 column
print(df1.loc[17057:17078, ['name', 'dob', 'gender', 'course']]) #First 2 rows, first 5 column
"""
results = Series([x.split()[0] for x in df1.city.dropna()])
#results = Series([x.split()[1] for x in df1.name.dropna()])

print(results)
print(results.value_counts()[:10])
cframe = df1[df1.course.notnull()]
print(cframe)
courselist = np.where(cframe['course'].str.contains('MSCDS'), 'MSCDS', 'PGDDS')
print(courselist[:])
#print(df.loc['R6':'R10', 'C':'E'])
by_course = cframe.groupby(['gender',courselist])
print(by_course)
agg_counts = by_course.size().unstack().fillna(0)
print(agg_counts)

indexer = agg_counts.sum(1).argsort()
print(indexer)

