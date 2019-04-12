# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:07:41 2018
Dhiraj
@author: du
http://scikit-learn.org/stable/modules/tree.html
"""

from sklearn import tree
X = [[0, 0], [1, 1]]
X
Y = [0, 1]
Y
clf = tree.DecisionTreeClassifier()
clf
clf = clf.fit(X, Y)
clf

clf.predict([[2., 2.]])

clf.predict_proba([[2., 2.]])


#
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
type(iris.data)
iris.data[0:1,::]
h,t=1,-1
iris.data[slice(h,t)]
h,t=None,None;
#For multidimensional arrays use a tuple of indices, which can include slice objects
iris.target
iris
str(iris)
clf = clf.fit(iris.data, iris.target)
clf
#conda install python-graphviz

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(clf, out_file=None, 
    feature_names=iris.feature_names,  
    class_names=iris.target_names,  
    filled=True, rounded=True,  
    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
clf.predict(iris.data[:1, :])
clf.predict_proba(iris.data[:1, :])