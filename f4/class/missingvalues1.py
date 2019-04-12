# -*- coding: utf-8 -*-
"""
Mon Mar 12 12:32:37 2018: Dhiraj
"""
import pandas as pd
import numpy as np
df = pd.read_csv('./data/dsstudents.csv')
df.columns
df

df.loc[ df['gender'] == 'M' , ['course','city'] ]

from scipy.stats import mode
mode(df['gender']).mode[0]  #more males
df

#create some missing values
df.columns
df.gender[df.rollno== 17057] = np.nan
df
df['gender']
df['gender'].notna()
pd.isna(df['gender'])
df['gender'].isna()
df
df.rpgm[df.course== 'PGDDS'] = np.nan
df
df['rpgm'].isna()

#count missing values
sum(df.isnull().values.ravel())
df.isnull().values.ravel().sum()

sum([True for ids, row in df.iterrows() if any(row.isnull())])

df.apply(lambda x: sum(x.isnull().values), axis=0) # columns
df.apply(lambda x: sum(x.isnull().values), axis=1) # rows
sum(df.apply(lambda x: sum(x.isnull().values), axis=1) >= 1) # rows with NA > 1
sum(df.apply(lambda x: sum(x.isnull().values), axis=1) >= 0) # rows with NA > 1

df.isnull().sum().sum()
#sum(map(any,df.isnull()))  rows with missing values

df.isnull().any(axis=1).sum()


df.isnull()
df.isnull().values.any()
df.isnull().sum()

df.dropna()
