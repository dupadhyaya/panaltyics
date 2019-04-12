# -*- coding: utf-8 -*-
"""
Mon Mar 12 11:33:13 2018: Dhiraj ; Histogram
"""
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = np.random.randn(1000)
plt.hist(data)
#more options
plt.hist(data, bins=30, normed=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')

# comparing distributions
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
x1[1:5]

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);


# Two Dim Histogram
mean = [0,0]
cov = [[1,1], [1,2]]
cov
x, y = np.random.multivariate_normal(mean, cov, 10000).T
type(x); x.size
x[1:5]
y[1:5]

plt.hist2d(x,y)
