# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:25:48 2019

@author: du
"""
#strings

import pandas as pd
studenturl = "https://raw.githubusercontent.com/dupadhyaya/panaltyics/master/data/studentdata.csv"
sdata = pd.read_csv(studenturl)
sdata
sdata.columns
sdata.head()
names = sdata.fullname
names
names.str.capitalize()
names.str.lower()
sdata.fullname.str.lower()
sdata.fullname.str.len()
sdata.fullname.str.startswith('AASHIMA')
sdata.fullname.str.split()
sdata.fullname.str.rsplit()
sdata.fullname.str.count('JOSHI*')
sdata.fullname.str.findall(r'^[AEIOU].*[^aieou]$') #start and with vowel

sdata.fullname.str[0:3].head()
sdata.fullname.str.get(-1).head()
