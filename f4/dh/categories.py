# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 08:20:27 2017 by Dhiraj Upadhyaya
"""

#Categories
import pandas as pd
from pandas import Series
import numpy as np

s = Series(["a","b","c","a"], dtype="category")
s
s.cat.categories
s.cat.ordered
type(s)

df = pd.DataFrame({'A':['a','b','c','a']})
df
df['B'] = df['A'].astype('category')
df
type(df['B'])


#pd.core.common.is_categorical_dtype(df.A.dtype)
df = pd.DataFrame({'value': np.random.randint(0, 100, 20)})
df
labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]



df = pd.DataFrame({'a' : [1, 2, 3, 4, 5], 'b' : ['yes', 'no', 'yes', 'no', 'absent']})
df['c'] = pd.Categorical.from_array(df.b).labels
df

pd.factorize(df.b)
df
df['c'] = pd.factorize(df.b)[0]
df

rollno = [17010,17045,17012, 17087,17057,17056,17084, 17078,17018,17013]
course = ['PGDDS', 'PGDDS','PGDDS','PGDDS','MSCDS', 'PGDDS', 'PGDDS','PGDDS', 'PGDDS','PGDDS']
gender = ['M','F','M','M','M','F','F','M','M','M']

sdata2= pd.DataFrame(list(zip(rollno,course,gender,)))
sdata2.columns=['rollno','course','gender']
sdata2
sdata2['coursecat'] = pd.factorize(sdata2.course)[0]
sdata2
sdata2['gendercat'] = pd.factorize(sdata2.gender)[0]
sdata2

#raw_cat = pd.Categorical(['PGDDS', 'PGDDS','PGDDS','PGDDS','MSCDS'], categories=["PGDDS","MSCDS"],ordered=False)
course1 = pd.Categorical(course, categories=["PGDDS","MSCDS"],ordered=True)
gender1 = pd.Categorical(gender, categories=('M','F'), ordered=True)
course1, gender1

s = pd.Series(course1, gender1)
s


s = pd.Series(["a", "b", "c", "a"])
s
from pandas.api.types import CategoricalDtype
CategoricalDtype(['a', 'b', 'c'])
CategoricalDtype(['a', 'b', 'c'], ordered=True)
