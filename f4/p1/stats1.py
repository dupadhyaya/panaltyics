# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 08:37:40 2017 by Dhiraj Upadhyaya
Stats
http://www.scipy-lectures.org/packages/statistics/index.html
"""

import pandas
data = pandas.read_csv('data/brain_size.csv', sep=';', na_values=".")
print(data.shape)
print(data)  
#observations of brain size and weight and IQ 
#If we don’t specify the missing value (NA = not available) marker, we will not be able to do statistical analysis.

import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
df2 = pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})
print(df2)
#pandas can input data from SQL, excel files, or other formats

print(data.shape)    # 40 rows and 8 columns
#(40, 8)

print(data.columns)  # It has columns   

print(data['Gender'])  # Columns can be addressed by name  
#Simpler selector
data[data['Gender'] == 'Female']['VIQ'].mean()
pandas.DataFrame.describe(df2) # large df
pandas.DataFrame.describe(data) # large df

groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
#('Female', 109.45) ('Male', 115.25)
print(groupby_gender.mean())  # Tabl on groupby_gender.

"""
What is the mean value for VIQ for the full population?
How many males/females were included in this study?
Hint use ‘tab completion’ to find out the methods that can be called, instead of ‘mean’ in the above example.
"""

from pandas.tools import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])  
#What is the average value of MRI counts expressed in log units, for males and females?
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']]) 
#Plot the scatter matrix for males only, and for females only. Do you think that the 2 sub-populations correspond to gender?


#3.1.2. Hypothesis testing: comparing two groups

from scipy import stats
print(stats.ttest_1samp(data['VIQ'], 0)) 

#3.1.2.1.2. 2-sample t-test: testing for difference across populations

female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
print(stats.ttest_ind(female_viq, male_viq))   

#3.1.2.2. Paired tests: repeated measurements on the same indivuals
print(stats.ttest_ind(data['FSIQ'], data['PIQ']))

stats.ttest_rel(data['FSIQ'], data['PIQ'])
#equivalent to a 1-sample test on the difference:
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)
stats.wilcoxon(data['FSIQ'], data['PIQ'])  


#3.1.3. Linear models, multiple factors, and analysis of variance
#3.1.3.1. “formulas” to specify statistical models in Python
#3.1.3.1.1. A simple linear regression

import numpy as np
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})
print(data)
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())  
#Retrieve the estimated parameters from the model above. Hint: use tab-completion to find the relevent attribute.

#3.1.3.1.2. Categorical variables: comparing groups or multiple categories
data = pandas.read_csv('data/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())

#Forcing categorical: the ‘Gender’ is automatically detected as a categorical variable, and thus each of its different values are treated as different entities.
#An integer column can be forced to be treated as categorical using:
model = ols('VIQ ~ C(Gender)', data).fit()  
 #Intercept: We can remove the intercept using - 1 in the formula, or force the use of an intercept using + 1.
#By default, statsmodels treats a categorical variable with K possible values as K-1 ‘dummy’ boolean variables (the last level being absorbed into the intercept term). This is almost always a good default choice - however, it is possible to specify different encodings for categorical variables 
# Link to t-tests between different FSIQ and PIQ
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)  
model = ols("iq ~ type", data_long).fit()
print(model.summary())

stats.ttest_ind(data['FSIQ'], data['PIQ']) 

#3.1.3.2. Multiple Regression: including multiple factors

data = pandas.read_csv('data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())  

#3.1.3.3. Post-hoc hypothesis testing: analysis of variance (ANOVA)
print(model.f_test([0, 1, -1, 0]))


#3.1.4. More visualization: seaborn for statistical exploration
