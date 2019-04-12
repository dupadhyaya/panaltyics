# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:21:06 2018 by Dhiraj Upadhyaya for DS 
"""

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
import pandas as pd
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df= pd.read_csv(url,header=None, names=names)
df.describe()
df.columns
df.head
import numpy as np
from sklearn.cross_validation import train_test_split

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:4]) 	# end index is exclusive
y = np.array(df['class']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X
y

# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)
pred, y_test
# evaluate accuracy
print(accuracy_score(y_test, pred))

#---------------------------
#https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4]) 	# end index is exclusive
y = np.array(df['class']) 	# another way of indexing a pandas df

# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)
pred
# evaluate accuracy
print(accuracy_score(y_test, pred))

# creating odd list of K for KNN
myList = list(range(1,50))
myList
# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)
type(neighbors)
list(neighbors)  # odd values

# empty list that will hold cv scores
cv_scores = []
#k=3
# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train,cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

cv_scores
# changing to misclassification error
MSE = [1 - x for x in cv_scores]
MSE
# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

import matplotlib as plt
# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()







# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#---- CV


import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred) #0.5
accuracy_score(y_true, y_pred, normalize=False)  #2

#multilabel cases
accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))



from scipy.spatial import distance
a = [1,2,3]
b = [4,5,6]
dst = distance.euclidean(a,b)
dst
# data 2
data1 = [2, 2, 2]
data2 = [4, 4, 4]
dist2 = distance.euclidean(data1, data2)
dist2

import numpy 
dist = numpy.linalg.norm(a,b)
