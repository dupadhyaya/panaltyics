# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:33:48 2018
Dhiraj
@author: du
"""
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

import numpy as np
iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
df.head()

from sklearn import datasets
import pandas as pd

boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df_boston['target'] = pd.Series(boston_data.target)
df_boston.head()
type(datasets.california_housing)
data= datasets.load_diabetes()
df = pd.DataFrame(datasets.load_diabetes().data)

df = pd.DataFrame(data.data)
df.describe()
df.mean()
list(df)
list(df.columns.value)
df.columns.values.tolist()
df.head()

data=load_iris()
items = load_iris            #Gets all the data from this Bunch - a huge list
type(items)
items
items[1][1]
mydata = pd.DataFrame(items[1][1])            #Gets the Attributes
mydata[len(mydata.columns)] = items[2][1]     #Adds a column for the Target Variable
mydata.columns = items[-1][1] + [items[2][0]] #Gets the column names and updates the 


import pandas as pd
#data = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
