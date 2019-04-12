# -*- coding: utf-8 -*-
"""
Groupby
"""
import pandas as pd
import numpy as np
students = pd.read_csv('students3.csv')
#students = pd.read_csv('students3.csv', dtype={'fees':float})
students.head()
students.columns
students.dtypes
students.select_dtypes(['object']) # only string
students['rollno'].dtype
students.index = students.rollno
students.drop(labels='Unnamed: 0',axis=1, inplace=True)
del students['Unnamed: 0']
students.describe()
students.groupby('course')['sclass'].describe()
students.groupby('course')['sclass'].describe().unstack()

students.groupby('sclass')  # nothing
students.groupby('sclass').aggregate(['min', np.median, max])
students[['sclass','python','sas']].groupby('sclass').aggregate(['min', np.median, max, np.sum, np.std])
students[['python']]

students.groupby('course').aggregate({'python': 'min', 'sas':'max'})


# Transformation
students[['course','python','sas']].groupby('course').transform(lambda x: x - x.mean())

# Apply
students[['course','python','sas']].groupby('course').apply( lambda x:x)

# Split Key
students.groupby(students['gender']).sum()


#apply
#DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
students.apply(np.sqrt) # returns DataFrame
students.apply(np.sum, axis=0) # equiv to df.sum(0)
students.apply(np.sum, axis=1) # equiv to df.sum(1)
