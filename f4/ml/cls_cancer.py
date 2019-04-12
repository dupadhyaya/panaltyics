# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:44:25 2018 by Dhiraj Upadhyaya for DS 
"""
import sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
type(data)  # Bunch

label_names = data['target_names']
label_names
labels = data['target']
labels
feature_names = data['feature_names']
features = data['data']
features[1:5]
#data.describe()
label_names
labels[:]
feature_names[0]
features[0]

from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
train

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
model
preds = gnb.predict(test)
preds

from sklearn.metrics import accuracy_score
accuracy_score(test_labels, preds)


#import statements can be moved up

