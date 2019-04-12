# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 06:28:03 2017 by Dhiraj Upadhyaya
"""
#Student Data 
import pandas as pd
from datetime import datetime
from dateutil.parser import parse

df = pd.read_csv('dsstudents.csv')
df
df.columns
df.dob
df['dob'].describe()
df['course'].describe()



#Date Column - Calculate Age
today = datetime.date.today() 
today
dob1 = datetime.datetime.strptime(df.dob, "%d-%b-%y").date()



dob1 = pd.to_datetime(df.dob)
type(dob1)
dob1
dob2 = [datetime.datetime.strptime(x, '%d-%b-%y') for x in df.dob]
type(dob2)
dob2
dob3 = [parse(x) for x in df.dob]
type(dob3)
dob3

#Diff
diffdays = (today - dob1)
age = diffdays/365
age
age.dt.days
import numpy as np
(age / np.timedelta64(1, 'D')).astype(int)
pd.to_timedelta(age)
#Age
df['age'] = age.dt.days
df
df.columns

df.describe()
df.drop(df.index[2], inplace=True)
df
#df = df.drop([5]).reset_index(drop=True)
df.index = range(len(df))
df

df.to_csv('dsstudents2.csv')
df.count()
#CSV file formated and saved

sdata2 = pd.read_csv('dsstudents2.csv')
sdata2.describe()
#Index(['Unnamed: 0', 'rollno', 'name', 'course', 'gender', 'hostel', 'dob','fees', 'email', 'mobno', 'city', 'rpgm', 'excel', 'sql', 'stats', 'age'],

sdata2a = sdata2.loc[:, lambda df: ['rollno', 'name', 'course', 'fees']]

sdata2a.columns
sdata2a
grouped1 = sdata2.groupby('course')
grouped1.sum()
grouped1.count()
sdata2a
sdata2a.iloc[[1,2,3],[1,2]]
sdata2a.iloc[:,1:2]
sdata2a.iat[1,1]
sdata2a.columns
sdata2a[sdata2.fees > 150000]
sdata2a[sdata2 > 150000]
sdata2a[sdata2a['course'].isin(['PGDDS','BSCDS'])]

#Missing Values
sdata2a.loc[sdata2a.fees > 15000, 'fees'] = np.nan 
sdata2a
sdata2a.dropna(how='any')
pd.isnull(sdata2a)
sdata2a.isnull().sum()
count_nan = len(sdata2a) - sdata2a.count()
count_nan
for col in sdata2a:
    print(sdata2a[col].value_counts(dropna=False))

sdata2a.isnull().any().any()
sum(pd.isnull(sdata2a['fees']))
sdata2a['fees'].value_counts(dropna = False)[np.nan]

sdata2.columns
sdata2b = sdata2.loc[:, lambda df: ['rollno', 'name', 'course', 'hostel','gender','fees','city', 'sql','stats','age']]
#Groups / Tables
group1 = sdata2b.groupby('course')
group1['sql', 'stats'].aggregate(np.sum)
group1['fees'].agg([np.sum, np.mean, np.std, len])
group1.mean()
group1.describe()


coursedata = sdata2b[['course','hostel','gender']].drop_duplicates()
coursedata

sdata2b.sample(n=4)
sdata2b.sort_values('rollno')
sdata2b.sort_values('name', ascending=False)['name']
sdata2b['sql'].apply(lambda x: x > 60)
sdata2b['course'].value_counts()
group2 = sdata2b.groupby(['course','gender'])
group2['fees'].value_counts()
group2.size()
group2.size().unstack()  #this is what was needed

#select course, COUNT(CASE WHEN fees > 150000) from sdata2b groupby course


coursewise = sdata2b.pivot_table(index='course', columns='gender', values='fees', aggfunc='sum')
coursewise

citywise = sdata2b.pivot_table(index='city', columns=['gender'], values='fees', aggfunc='sum')
citywise

sdata2b.pivot_table(columns='fees')
sdata2b.pivot_table(index='course', columns='fees',values='stats', aggfunc='mean')

sdata2b.pivot_table(index='course',values='stats', aggfunc='mean')
