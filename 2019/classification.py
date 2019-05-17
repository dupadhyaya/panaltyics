# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:36:43 2019

@author: du
"""
#classification

from sklearn import datasets
iris = datasets.load_iris()
iris


from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target) # learn from the data 

clf.predict([[ 5.0,  3.6,  1.3,  0.25]])

from sklearn import svm
svc = svm.SVC(kernel='linear')

svc.fit(iris.data, iris.target) 
svc = svm.SVC(kernel='linear')
svc

# Create and fit a nearest-neighbor classifier
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target) 

knn.predict([[0.1, 0.2, 0.3, 0.4]])


from sklearn import cluster, datasets
iris = datasets.load_iris()
cluster.KMeans?
k_means = cluster.KMeans(k=3)
k_means.fit(iris.data) 

print k_means.labels_[::10]

print iris.target[::10]


#PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)

X = pca.transform(iris.data)
X
import pylab as pl
pl.scatter(X[:, 0], X[:, 1], c=iris.target)
