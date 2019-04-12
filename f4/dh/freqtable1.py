# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:32:09 2017 by Dhiraj Upadhyaya
"""

#Frequency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Eg1
series1 = pd.Series([1,2,2,3,3,3])
series1.value_counts()

#Eg2
series2 = pd.Series(['a','b','a','c','b'])
series2.value_counts()

#Method3

vals, counts = np.unique(series2, return_counts=True)
vals, counts
results = dict(zip(vals, counts))
results
pd.Series(results)


#Method4
df = pd.read_csv('dsstudents2.csv')
df
tab1 = pd.crosstab( index=df["course"], columns="counts")
tab1
tab1/tab1.sum()
tab1.hist()
y=tab1['counts'].values
x=range(len(tab1.index))
x
y
plt.bar(x,y, align='center', alpha=0.5)
plt.show()




tab2 = pd.crosstab(index= df["course"],columns= df['gender'], margins=True)
tab2
tab2.index=['Masters','Diploma','ColTotal']
tab2.columns=['Female','Male','RowTotal']
tab2
tab2/tab2.ix['ColTotal','RowTotal']
#proportion of counts along each column
tab2/tab2.ix['ColTotal']
tab2.div(tab2['RowTotal'],axis=0)
tab2.T/tab2["RowTotal"]


tab3 = pd.crosstab(index=df['city'], columns= df['course'])
tab3
tab3.columns=['Masters','Diploma']
tab3

#Higher Dimensional Tables
tab5 = pd.crosstab(index=df["course"],                            columns=[df["hostel"],df["gender"]], margins=True)
tab5
tab5[0]
tab5[1]
tab5[1].F
tab5[1]['M']
tab5/tab5.loc["All"]
tab5.shape

