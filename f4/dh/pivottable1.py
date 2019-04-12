# -*- coding: utf-8 -*-
"""
Thu Feb 15 15:03:25 2018: Dhiraj
"""
#%% Student Data
import numpy as np
import pandas as pd
rollnoL = [109,102,105,106,103,110,101,107,104,111,108]
nameL = ['meena','apoorva','kastav','shubam', 'goldie', 'hitesh', 'shruti', 'vijay','achal', 'lalit', 'varun']
courseL= ['pg','pg','msc', 'msc','pg','pg','pg', 'pg','pg','pg','bsc']
genderL =['F','F','M','M','M','M','F','M','M','M','M']
pythonL = np.random.randint(60,90,11)
sasL = np.random.randint(65,85,11)
hadoopL = np.random.randint(71,90,11)
feesL = np.random.randint(100000, 150000, 11)
hostelL = [True, False, True, False, False, True, False, True, True, True, False]

rollnoS = pd.Series(rollnoL)
nameS = pd.Series(nameL)
genderS = pd.Series(genderL)
pythonS = pd.Series(pythonL)
sasS = pd.Series(sasL)
hadoopS = pd.Series(hadoopL)
feesS = pd.Series(feesL)
#hostelS = pd.Series([True, False, True, False, False, True, False, True, True, True, False])
hostelS = pd.Series(hostelL)
feesS = pd.Series(feesL)
courseS = pd.Series(courseL)

studentDF1 = pd.DataFrame({'rollno':rollnoL,'name':nameL, 'gender':genderL, 'python':pythonL, 'sas':sasL, 'hadoop':hadoopL, 'fees':feesL, 'course':courseL}, columns = ['rollno','name', 'gender', 'python', 'sas', 'hadoop', 'fees', 'course'])
studentDF1
studentDF1.index = rollnoL
studentDF1

studentDF2 = pd.concat([rollnoS, nameS, genderS, hostelS, pythonS, sasS, hadoopS, feesS, courseS], axis=1)
studentDF2
studentDF2.columns = ['rollno', 'name', 'gender', 'hostel', 'python', 'sas', 'hadoop', 'fees', 'course']
studentDF2
# save to csv

studentDF2['total'] = studentDF2['python'] + studentDF2['sas'] + studentDF2['hadoop']
studentDF2

studentDF2.to_csv("students2.csv")

# Pivot Table
studentDF2.columns
studentDF2.groupby('gender').mean()
studentDF2.groupby('gender').size()
#studentDF2.groupby('gender').filter(lambda gender: gender.size > 8)

studentDF2.groupby(['gender','course'])['hostel' ].aggregate( 'mean' ).unstack()
# similar output
studentDF2.pivot_table('hostel', index='gender', columns='course')

for i in range(len(studentDF2)):
    print(type(studentDF2.iloc[i]), end=' ')
    
studentDF2['total'].describe()
studentDF2.describe()
pythonB = pd.cut(studentDF2['python'],[0, 70, 100])
pythonB
studentDF2
studentDF2.pivot_table('hostel',['gender', pythonB], 'course')
studentDF2.pivot_table('hostel',['gender', total], 'course')
#http://pbpython.com/pandas-pivot-table-explained.html

#Random values for a column
from numpy import random

classes = ['C1', 'C2', 'C3']       
sclass = random.choice(classes, 11)
sclass
studentDF2['sclass'] = pd.Series(sclass)
studentDF2
studentDF2.to_csv("students3.csv")
studentDF2.columns
pd.pivot_table(studentDF2, index=['name'])
pd.pivot_table(studentDF2, index=['name', 'sclass','hostel'])
pd.pivot_table(studentDF2, index=['sclass','gender'])
# smart to start aggregating data and grouping 
pd.pivot_table(studentDF2, index=['course','sclass'], values=['total','python'])
# default is mean
pd.pivot_table(studentDF2, index=['course','sclass'], values=['total','python'], aggfunc=np.sum)
pd.pivot_table(studentDF2, index=['course','sclass'], values=['total','python'], aggfunc=[np.sum, np.mean, len])

pd.pivot_table(studentDF2, index=['gender','course'], values=['fees'], columns=['sclass'], aggfunc=[np.sum])
# missing values will cause NaN
pd.pivot_table(studentDF2, index=['gender','course'], values=['fees', 'total'], columns=['sclass'], aggfunc=[np.sum], fill_value=0)

pd.pivot_table(studentDF2, index=['gender','course'], values=['fees', 'total'], columns=['sclass'], aggfunc=[np.sum], fill_value=0, margins=True)

pd.pivot_table(studentDF2, index=['gender','hostel'], values=['fees'], aggfunc=[np.sum], fill_value=0, margins=True)

pd.pivot_table(studentDF2, index=['gender','course'], values=['fees', 'total'], columns=['sclass'], aggfunc={'fees':np.sum, 'total':np.max}, fill_value=0)

table = pd.pivot_table(studentDF2, index=['name', 'course'], values=['fees','total'], aggfunc={'fees':np.sum, 'total':np.max}, fill_value=0)
table
table.query("name ==['meena','apoorva']")
table.query("course ==['pg','bsc']")



# Pivot
studentDF2.pivot(index='name', columns='sclass', values='total')

# 
import datetime
len([datetime.datetime(2018,1,i) for i in range(1,12)])
