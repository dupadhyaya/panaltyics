# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:01:55 2018
Dhiraj
@author: du
"""

import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library 
data
type(data)
data.DESCR
import numpy as np
import pandas as pd
# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
type(target)
target.head()
# Without a constant

import statsmodels.api as sm

X = df["RM"]
X
y = target["MEDV"]
y
# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
