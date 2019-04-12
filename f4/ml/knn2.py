# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:23:14 2018 by Dhiraj Upadhyaya for DS 
"""

from sklearn.datasets import load_iris
from sklearn import cross_validation
import numpy as np
 
# load dataset and partition in training and testing sets
iris = load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=1)
X_train, y_train
 
# reformat train/test datasets for convenience
train = np.array(zip(X_train,y_train))
test = np.array(zip(X_test, y_test))
a=zip([1,2,3], [4,5,6])

import math
 
# 1) given two data points, calculate the euclidean distance between them
def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))

train[0,0]
get_distance(train[0][0], train[1][0])
get_distance([1,1], [2,3])
