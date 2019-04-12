# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:50:33 2018
Dhiraj
@author: du
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston
http://scikit-learn.org/stable/datasets/index.html
"""

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)