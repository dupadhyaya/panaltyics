# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:46:45 2017 by Dhiraj Upadhyaya
CSV File Read / Write
"""
""
import csv
with open('./data/dsstudents.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        print(', '.join(row))
"""

import pandas as pd
#df1 = pd.read_excel('file.xls', skiprows=17)
#print(df)

import pandas as pd
df = pd.read_csv('./data/dsstudents.csv')
print(df)
