# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:51:38 2018
Dhiraj
@author: du
"""

#Data sets
from rpy2.robjects import r, pandas2ri
def data1(name): 
    return pandas2ri.ri2py(r[name])

df = data1('iris')
df.describe()

df = data1('mtcars')
df
df.describe()
df2 = data1('mtcars')
df2.describe()
print(r.data())
df3=pandas2ri.ri2py(r('women'))
df3

#$ pip install pydataset
#then just load up any dataset you wish (currently around 757 datasets available) :

from pydataset import data

titanic = data('titanic')
titanic

from sklearn import datasets

from sklearn import datasets
iris = datasets.load_iris()
iris
type(iris)
iris.data


import seaborn as sns
iris = sns.load_dataset('iris')
iris

from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
column_names = iris.feature_names
column_names
iris.data
import pandas as pd
df = pd.DataFrame(iris.data, iris.feature_names)


iris = load_iris()
print(iris)
iris.describe


from statsmodels.datasets import Dataset
d = load() # -> d is a Dataset object, see below

from statsmodels import datasets
data = datasets.longley.load()
data.data
