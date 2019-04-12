#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


column_names = ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
data=pd.read_csv("C:\\Users\\acer\\Desktop\\Internship data\\bank-additional\\bank-additional\\bank-additional-full.csv",header=0,names=column_names,sep=";")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data["education"].unique()


# In[7]:


data["y"].value_counts() ###so,this is a highly biased dataset because the most of the cases are no and few are yes


# In[8]:


value_counts = data['y'].value_counts()

value_counts.plot.bar(title = 'y value counts')


# In[9]:


count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


# In[10]:


data.groupby("y").mean()


# In[11]:


data.groupby("education").mean()


# In[12]:


data.groupby("marital").mean()


# In[13]:


#####Visualization#####
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


# In[14]:


table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')


# In[15]:


table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')


# In[16]:


pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')


# In[17]:


pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')


# In[18]:


data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')


# In[19]:


pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')


# In[21]:


data = data[['age','job','marital','loan','education','default','housing','y']]


# In[23]:


data.tail()


# In[25]:


data = pd.get_dummies(data,drop_first=True)


# In[26]:


data.info()


# In[29]:


###Creating dependent and independent variable
X = data.iloc[:,0:28].values
y = data.iloc[:,28].values.reshape(-1,1)


# In[30]:


####training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[33]:


####predicting the test result
y_pred = classifier.predict(X_test)


# In[34]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[35]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[36]:


print("Report:",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))


# In[39]:


score=classifier.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size=15);


# In[40]:


####Model->2
###SVM
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'sigmoid', random_state = 0)
classifier3.fit(X_train, y_train)


# In[41]:


# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)


# In[42]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)


# In[43]:


print("Report:",classification_report(y_test,y_pred3))
print("Accuracy:",accuracy_score(y_test,y_pred3))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred3))


# In[44]:


score3=classifier3.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm3,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score3)
plt.title(all_sample_title,size=15);


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[47]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[48]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[49]:


print(classification_report(y_test, y_pred))


# In[50]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:




