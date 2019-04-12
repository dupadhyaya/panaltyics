# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:05:35 2017 by Dhiraj Upadhyaya
"""

#idioms
import pandas as pd
import numpy as np
marks0 = pd.DataFrame({'unit1':[4,5,6,7],'unit2': [10,11,15,27], 'unit3':[56, 61,65, 68]})
marks0
marks = marks0.abscopy

marks.loc[marks.unit1 < 6, 'unit3' ] = -1 ; marks
marks
marks.loc[marks.unit1 > 6, ['unit2','unit3' ] ]= 100 ; marks

#same coln names
dfmask = pd.DataFrame({'unit1':[True] * 4,'unit2':[False]*4, 'unit3':[True, False] * 2})
dfmask

#display values from other df as per mask
marks
marks.where(dfmask,-10)
marks.where(dfmask,other=-100)

marks= marks0.copy()
marks['grade'] = np.where(marks['unit3'] > 60, 'high','low' ) ; marks


#Splitting
marks0
marks1 = marks0.copy()
lowmarks = marks1[marks1.unit1 <= 5]
lowmarks
marks0
highmarks = marks1[marks1.unit1 > 5]
highmarks

#Criteria Based
marks2 = marks0.copy()
newmarks = marks2.loc[(marks2['unit1'] > 5) & (marks2['unit3'] > 60), 'unit2' ] ; newmarks 
marks2
#other or | , inplace
marks2.loc[(marks2['unit1'] > 5) | (marks2['unit3'] > 60), 'unit2' ] = 0.1 ; marks2 
marks0
#xxx
marks3= marks0.copy()
crit1 = marks3.unit1 <= 6
crit2 = marks3.unit2 = 15
crit3 = marks3.unit3 > 60
allcrit = crit1 & crit2 & crit3 
critlist = [crit1, crit2, crit3]
import functools
allcrit2 = functools.reduce(lambda x,y : x & y, critlist)
marks3[allcrit]
marks3[allcrit2]


#Selection
marks4= marks0.copy()
marks4[(marks2.unit1 <=6) & (marks4.index.isin([0,2,3]))]
marks4

marks5 = marks0.copy()
marks5.index = ['s1','s2','s3','s4']
marks5
marks5.loc['s1']   #label
marks5.loc[['s1','s3']]
marks5.loc['s1':'s3']

marks5.iloc[0]  #position
marks5.iloc[0,2] #particular cell
marks5.iloc[0:2] 

marks5[['unit1','unit2']]
marks5['unit1']

marks5.iloc[1:-2]

marks4[~((marks4.unit1 <=6) & (marks4.index.isin([0,2])))]
