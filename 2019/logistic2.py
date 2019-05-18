# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:09:22 2019 @author: du
"""
#https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

#import pandas
import pandas as pd
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
data=pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None, names=col_names)
#pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)
data.head()
pima=data
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable
X
y
X.shape
y.shape
#split
# split X and y into training and testing sets
#pip install sklearn in anconda prompt
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split#renamed/ depreciated
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
X_train
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
#https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
#logreg = LogisticRegression()
# create and configure model
#logreg = LogisticRegression(solver='lbfgs')
logreg = LogisticRegression(solver='liblinear')
# fit the model with data
logreg.fit(X_train,y_train)

#predict
y_pred=logreg.predict(X_test)
y_pred

#confusion matrix
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

#visualising
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
Text(0.5,257.44,'Predicted label')

#accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#ROCR curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
