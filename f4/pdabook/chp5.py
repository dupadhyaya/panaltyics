# -*- coding: utf-8 -*-
"""
Fri Feb  9 11:51:16 2018: Dhiraj
Chapter - 5
"""

#%% Page 343
#data as table
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()


#target array

%matplotlib inline
import seaborn as sns
sns.set()
iris.shape
sns.pairplot(iris, hue='species', size=1.5);
X_iris = iris.drop('species', axis=1)
X_iris.head()
X_iris.shape
y_iris = iris['species']
y_iris.head()
y_iris.shape
type(X_iris), type(y_iris)


#%% Supervised Learning Eg
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.RandomState(50)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
#diff rng.rand & rng.randn
plt.scatter(x,y)

#1. Choose class of Model
from sklearn.linear_model import LinearRegression

#2. Choose Parameters
model = LinearRegression(fit_intercept=True)
model

#Arrange data into a feature matrix & target vector
x[0:5]
x.shape
X = x[:, np.newaxis]
X.shape  #another blank col added
X[0:5,:]

#4. Fit the model to data
model.fit(X,y)
model.coef_
model.intercept_

#5. Predict
xfit = np.linspace(-1,11)
xfit
#convert them into feature matrix
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
yfit

plt.scatter(x,y)
plt.plot(xfit, yfit)
