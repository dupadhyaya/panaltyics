# -*- coding: utf-8 -*-
"""
Mon Feb 19 16:01:18 2018: Dhiraj
"""


import pandas as pd
import numpy as np
students = pd.read_csv('students3.csv')
students
del students['Unnamed: 0']
students
aggregations = {'python': {'total_python':'sum', 'avg_python':'mean'},         'sas': {'total_sas':'sum', 'avg_sas':'mean'},'fees': {'max_fees': 'max','min_fees': 'min'}, 'gender': ['count']}

students.groupby('sclass').agg(aggregations)

grouped2= students.groupby('course').agg({'python':[min, max, np.mean], 'sas':[min, max, np.mean]})
grouped2
grouped2.columns = ["_".join(x) for x in grouped2.columns.ravel()]
grouped2

grouped3 = students.groupby(['course','gender']).agg({'fees':[min, max]})
grouped3
grouped3.columns = grouped3.columns.droplevel(level=0)
grouped3
grouped3.rename(columns = {'min':'min_fees', 'max':'max_fees'}, inplace=True)
grouped3

students.groupby('course', as_index=False).agg({'fees':'sum'})
students.groupby('course', as_index=True).agg({'fees':'sum'})

students.groupby(['course','gender','hostel'])['rollno'].count()
students.groupby(['course'])['rollno'].first()

students['course'].value_counts()
students['course'].count()

students.groupby(['course']).groups.keys()
students['course'].nunique()   #not null entries
students['name'].nunique()   #not null entries

students['course'].unique()
