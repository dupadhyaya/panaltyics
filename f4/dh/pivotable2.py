# -*- coding: utf-8 -*-
"""
Thu Feb 22 09:31:20 2018: Dhiraj
"""

# Multi Dim Summarisation using Pivot Table2
import pandas as pd
import numpy as np
studentDF = pd.read_csv('students3.csv')
studentDF
del studentDF['Unnamed: 0']
studentDF

# Simple Pivot
studentDF.columns
pd.pivot_table(studentDF, index = 'course', values='sas')
pd.pivot_table(studentDF, index = ['course','gender'], values='sas')
pd.pivot_table(studentDF, index = ['course','gender'], columns='sclass', values=['sas','hadoop'])

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.pivot_table(studentDF, index='gender', columns='sclass', values='sas').plot(kind='bar')


#Aggregate Function
pd.pivot_table(studentDF, index='course', values=['sas','hadoop'], aggfunc = [np.mean, np.median, min, max])
pd.pivot_table(studentDF, index='course', values=['sas','hadoop'], aggfunc = [min, max, lambda x:x.count()])

# multiple Statistics per Group
aggregation = { 'sas' : { 'totalsas' : 'sum', 'avgsas':'mean'}, 'hadoop': {'meanhadoop':'mean', 'stdhadoop':'std'}}
studentDF[studentDF['sclass']=='C1'].groupby('gender').agg(aggregation)




lalit1 = { 'sas' : { 'totalsas' : ['sum','mean']}}
studentDF.groupby('gender').agg(lalit1)



studentDF.groupby(['gender','sclass']).agg({'python':[min,max,np.mean]})
