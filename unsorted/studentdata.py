# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 08:24:48 2017 by Dhiraj Upadhyaya
"""
import pandas as pd
# Course Data
rollno = [17010,17045,17012,17087,17057,17056,17084,17078,17018,17013]
sname =  ['Achal Kumar','Apoorva Karn','Goldie Sahni','Hitesh Mann','Kaustav Chatterjee','Meena Srisha Valavala','Shruti Sinha','Shubham Pujan','Vijay Pal Singh','Lalit Sahni']
course = ['PGDDS', 'PGDDS','PGDDS','PGDDS','MSCDS', 'PGDDS','PGDDS','PGDDS', 'PGDDS','PGDDS']
gender = ['M','F','M','M','M','F','F','M','M','M']
exp =  [3,3.5,14,5,0,1,15,1,3,8] # Experience
hostel =  [False,False,False,False,False,True,False,False,True,True]
rollno
sname
course
gender
hostel
len(hostel)
type(gender)

sdata = pd.DataFrame({'rollno':[rollno], 'sname':[sname], 'course':[course],'gender':[gender],'hostel':[hostel]}, columns=['rollno','sname','course','gender','hostel'])

sdata


sdata2= pd.DataFrame(list(zip(rollno,sname,course,gender, hostel)))
sdata2
sdata2.columns=['rollno','sname','course','gender','hostel']
sdata2

sdata2a = pd.DataFrame(list(map(list, zip(rollno,sname,course))))
sdata2a.columns= ['rollno','sname','course']
sdata2a

import numpy as np
pd.DataFrame(np.column_stack([rollno,sname,course]), columns=['rollno','sname','course'])



rollno = [17010,17045,17012,17087,17057]
sname =  ['Achal','Apoorva','Goldie','Hitesh','Kaustav']
course = ['PGDDS', 'PGDDS','PGDDS','PGDDS','MSCDS']
gender = ['M','F','M','M','M']
exp =  [3,3.5,14,5,0]
hostel = [True,False,False,True,True]

student =[{'rollno':17010, 'sname':'Achal', 'course':'PGDS', 'gender':'M','exp':3}, {'rollno':17045, 'sname':'Apoorva', 'course':'PGDS', 'gender':'F','exp':3}]

student
df = pd.DataFrame(student,columns=['gender','exp','rollno', 'sname']) ;df

df.index = df['rollno']
df
student2 = {'sname': ['Achal', 'Apoorva'], 'rollno': [17010, 17045],'gender': ['M', 'F'],'exp': [3,3]}
df2 = pd.DataFrame.from_dict(student2)
df2
df2 = df2[['rollno', 'sname','gender', 'exp']]
df2

#preserve ordr of keys
from collections import OrderedDict
student3 = OrderedDict([ ('sname', ['Achal', 'Apoorva',]),('rollno', [17010, 17045]),('gender',['M','F']),('exp', [3, 3]) ] )
student3
df3 = pd.DataFrame.from_dict(student3)
df3

student3a = OrderedDict([ ('rollno', [17010, 17045]),('sname', ['Achal', 'Apoorva',]),('gender',['M','F']),('exp', [3, 3]) ] )
df3a = pd.DataFrame(student3a)
df3a


#From Lists
student4=[(17010,'Achal','M',3),(17045,'Apoorva','F',3)]
labels = ['rollno', 'sname', 'gender', 'exp']
df4 = pd.DataFrame.from_records(student4, columns=labels)
df4
#from ordereddict
student5 = [('rollno', [17010, 17045]),('sname', ['Achal', 'Apoorva',]),('gender',['M','F']),('exp', [3, 3]) ] 
df5 = pd.DataFrame.from_items(student5)
df5

#http://pbpython.com/pandas-list-dict.html

