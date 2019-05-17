# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:08:25 2019

@author: du
"""
#datasets
#pip install pydataset
import rpy2
from rpy2.robjects import r, pandas2ri
def data(name): 
    return pandas2ri.ri2py(r[name])
#not working


#$ pip install pydataset
    
from pydataset import data

titanic = data('titanic')
titanic

#
from sklearn import datasets

load_boston() #         Load and return the boston house-prices dataset (regression).
load_iris()  #          Load and return the iris dataset (classification).
load_diabetes()#        Load and return the diabetes dataset (regression).
load_digits([n_class]) #Load and return the digits dataset (classification).
load_linnerud() #       Load and return the linnerud dataset (multivariate regression).



from sklearn import datasets
iris = datasets.load_iris()
iris
iris.data.shape
iris.target.shape
import numpy as np
np.unique(iris.target)
#----------
import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()


#-------
import pandas as pd
iris2 = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
iris2.head()

#---------
#R sample datasets
#Since any dataset can be read via pd.read_csv(), it is possible to access all R's sample data sets by copying the URLs from this R data set repository.
#Additional ways of loading the R sample data sets include statsmodel

import statsmodels.api as sm

iris = sm.datasets.get_rdataset('iris').data
and PyDataset

from pydataset import data

iris = data('iris')

#-----
#Quilt Quilt is a dataset manager created to facilitate dataset management. It includes many common sample datasets, such as several from the uciml sample repository. The quick start page shows how to install and import the iris data set:

# In your terminal
#$ pip install quilt
#$ quilt install uciml/iris
#After installing a dataset, it is accessible locally, so this is the best option if you want to work with the data offline.

import quilt.data.uciml.iris as ir

iris = ir.tables.iris()

