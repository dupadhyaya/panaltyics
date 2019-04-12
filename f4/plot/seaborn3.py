# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:53:04 2017 by Dhiraj Upadhyaya
"""
#seaborn 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set
data = np.random.multivariate_normal([0,0],[[5,2],[2,2]], size=2000)
data
data = pd.DataFrame(data, columns=['x','y'])
data

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)

for col in 'xy':
    sns.kdeplot(data[col], shade=True)

sns.distplot(data['x'])
sns.distplot(data['y'])

sns.kdeplot(data)

with sns.axes_style('white'):
    sns.jointplot('x','y', data, kind='kde')

with sns.axes_style('white'):
    sns.jointplot('x','y', data, kind='hex')


#Pair Plots
iris = sns.load_dataset('iris')
iris.head()
sns.pairplot(iris, hue='species', size=2.5)
