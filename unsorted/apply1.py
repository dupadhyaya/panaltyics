# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:27:17 2017 by Dhiraj Upadhyaya
"""

#apply
import pandas as pd
import numpy as np
df = pd.read_csv('./data/dsstudents2.csv')
df.columns
cols = ['course', 'gender', 'rpgm', 'excel', 'sql', 'stats']
df2 = df[cols]
df2
grouped = df.groupby('course')
grouped['gender'].apply(lambda x: x.describe())

grouped = df.groupby('course')['excel']
def f(group): return pd.DataFrame({'original' : group, 'demeaned' : group - group.mean()})
grouped.apply(f)

def f(x): return pd.Series([ x, x**2 ], index = ['x', 'x^2'])

s.apply(f)
