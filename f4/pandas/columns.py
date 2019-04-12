# -*- coding: utf-8 -*-
"""
Tue Feb 13 11:39:23 2018: Dhiraj
"""

#%% Managing Columns in DF

sDF = pd.read_csv("students1.csv")
sDF
sDF.columns
sDF.columns[::-1]  # reverse column
sDF[sDF.columns[::-1]]

hostelL=[True, False, True, False, False, True, False, True, True, True, False]
hostelS = pd.Series(hostelL)
hostelS
sDF2= pd.concat([sDF, hostelS], axis=1)
sDF2.rename(columns={0:'hostel'}, inplace=True)
sDF2.columns
sDF2.hostel.count()
sDF2.hostel.value_counts()
sDF2.isna()
sDF2.isnull().sum()




