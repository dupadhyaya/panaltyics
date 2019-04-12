# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:29:20 2017 by Dhiraj Upadhyaya
"""
#pandas 6
#Pg 170 Pivot Tables
import numpy as np
np.set_printoptions(precision=4)
print(np.pi * np.arange(8))
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic
titanic.head()
titanic.iloc[1:4,1:8]
titanic.loc[1:5, ('sex','class','age','survived')]
titanic.groupby('sex')[['survived']].mean()
titanic.groupby(['sex','class'])[['survived']].aggregate('mean').unstack()

#pivot table
titanic.pivot_table('survived', index='sex', columns='class')

age = pd.cut(titanic['age'],[0,18,80])
age
titanic['age']
titanic.pivot_table('survived', ['sex', age], 'class')
titanic.loc[1:5, ('sex','class','age','survived','fare')]
fare = pd.qcut(titanic['fare'],2)
fare
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

titanic.pivot_table(index='sex', columns='class', aggfunc={'survived':sum, 'fare':'mean'})

titanic.pivot_table('survived', index='sex', columns='class', margins=True)

titanic.pivot_table('survived', index='sex', columns='class', margins=True, aggfunc=sum)
#titanic.pivot_table('survived', index='sex', columns='class', margins=True, aggfunc={'sum','max'})

#Birthrate Data
births = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv')
births
births.head()
births.columns
births['decade'] = 10 * (births['year']) // 10
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')

%matplotlib inline
import matplotlib.pyplot as plt
sns.set()
births.pivot_table('births', index='year', columns = 'gender', aggfunc='sum').plot()
plt.ylabel('total births per year');


#Further data exploration
quartiles = np.percentile(births['births'], [25, 50, 75])
quartiles
mu = quartiles[1]
sig = 0.74 * quartiles[2] - quartiles[0]
sig
births
births['births'].head()

mu, sig

births2 = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
#births2 = births.query('births < @mu + 5 * @sig')

births2
