# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:09:24 2017 by Dhiraj Upadhyaya
"""

#Index
import pandas as pd
import numpy as np

df = pd.read_csv('dsstudents2.csv')
df.columns
cols = ['course', 'gender', 'fees', 'city', 'rpgm', 'excel', 'sql', 'stats']
df2 = df[cols]
df2

grouped1 = df2.groupby('course')
grouped2 = df2.groupby(['course', 'gender'])
#pandas Index objects support duplicate values.
grouped1.first()
grouped1.last()
grouped1.sum()
grouped1.groups
grouped1.rank()

grouped2.first()
grouped2.last()
grouped2.sum()
round(grouped2.mean(),1)
grouped2.groups
len(grouped2)
grouped2.min()
grouped2.max()
grouped2.count()
grouped2.aggregate(sum)
grouped2.describe()

grouping = [1, 2, 3, 1, 2, 3]
marks = pd.Series([50, 57, 63, 67, 50, 70], grouping)
grouped3 = marks.groupby(level=0)
grouped3.first()
grouped3.last()
grouped3.sum()
grouped3.mean()

#https://pandas.pydata.org/pandas-docs/stable/groupby.html

#Sorting in groups
df2.groupby(['course']).sum()
round(df2.groupby(['course','gender']).mean(),1)
df2.groupby(['city'], sort=False).sum()
df2.groupby(['city'], sort=True).sum()

df2.groupby(['city']).get_group('Noida')
df2.groupby(['city']).groups




#Muti Index
arrays = [['ds', 'ds', 'ds', 'ds', 'mls', 'mls', 'mls', 'anm','anm','anm'], ['pg', 'msc', 'bsc', 'btech', 'ballb', 'llb', 'bcomllb', 'bsc','msc','ma']]
arrays
index = pd.MultiIndex.from_arrays(arrays, names=['dept', 'course'])

s = pd.Series(np.random.randint(10,20,10), index=index)
s
grouped0 = s.groupby(level=0)
grouped0.sum()

grouped1 = s.groupby(level=1)
grouped1.sum()

s.groupby(level='dept').sum()
s.groupby(level='course').sum()

s.sum(level='dept')
s.sum(level='course')

s


dept = ['ds', 'ds', 'ds', 'ds', 'mls', 'mls', 'mls', 'anm','anm','anm']
course = ['pg', 'msc', 'bsc', 'btech', 'ballb', 'llb', 'bcomllb', 'bsc','msc','ma']
sem= ['sem1' ,'sem3' ,'sem5','sem1' ,'sem1' ,'sem3' ,'sem1' ,'sem1','sem1','sem1']

index2 = pd.MultiIndex.from_arrays([dept, course, sem], names=['dept', 'course','sem'])
s2 = pd.Series(np.random.randint(10,20,10), index=index)
s2

s2.groupby(level=['dept', 'sem']).sum()

s2.groupby(['course', 'sem']).sum()


#Grouping DataFrame with Index Levels and Columns¶

df = pd.DataFrame({'floor': [3, 3, 3, 3, 2, 2, 2, 1,1,1], 'students': np.arange(10)}, index=index)
df
df.groupby([pd.Grouper(level=0), 'floor']).sum()
df.groupby([pd.Grouper(level=0), 'students']).sum()
df.groupby([pd.Grouper(level=1),'floor']).sum()
df.groupby([pd.Grouper(level='dept'), 'floor']).sum()
df.groupby([pd.Grouper(level='dept'), 'students']).sum()
df.groupby([pd.Grouper(level='sem'), 'floor']).sum()


#DataFrame column selection in GroupBy
grouped = df.groupby(['dept'])
grouped.sum()
grouped_c = grouped['floor']
grouped_c.sum()

grouped_D = grouped['students']
grouped_D.sum()

grouped2 = df.groupby('dept')
grouped2.get_group('ds')
grouped2.get_group('anm')

df.groupby(['dept', 'course']).get_group(('ds', 'bsc'))
df.groupby(['dept', 'course']).get_group(('ds', 'msc'))


#Aggregation aggregate or equivalently agg method:
grouped3 = df.groupby('dept')
grouped3.aggregate(np.sum)

grouped4 = df.groupby(['dept', 'course'])
grouped4.agg(np.sum)

grouped5 = df.groupby(['dept'], as_index=False)
grouped5.aggregate(np.sum)
df.groupby('course', as_index=False).sum()
df.groupby('course', as_index=True).sum()

grouped5.size()
grouped5.describe()

grouped5['floor'].agg([np.sum, np.mean, np.std])
grouped5['students'].agg([np.sum, np.mean, np.std])
grouped5.agg([np.sum, np.mean, np.std])

#Applying different functions to DataFrame columns
grouped5.agg({'students' : np.sum,'floor' : lambda x: np.std(x, ddof=1)})

import collections, OrderedDict
grouped5.agg({'students' : 'sum', 'floor' : 'mean'})


#Transformation


#Apply
