# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:06:59 2017 by Dhiraj Upadhyaya
Random Nos and Plot  Correlation and Covariance
"""
import numpy as np

#returns integers with a discrete uniform distribution
x1= np.random.randint(-10, 10)

#returns floats with a normal distribution
x2= np.random.normal(0, 0.1, 1)
print(x1, x2)

x3 = np.random.normal(loc=0.0, scale=1.0, size=10)
print(x3)

mu, sigma = 60, 10 # mean and standard deviation
s = np.ceil(np.random.normal(mu, sigma, 1000))
print(s)
print(len(s))
#print(np.round_(s))
print(np.std(s))
print(np.mean(s))

abs(mu - np.mean(s)) < 0.5
#True

abs(sigma - np.std(s, ddof=1)) < 0.5
#True - checking values of SD given and calc
print(sigma, np.std(s), sep=' ')

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()
print(count, bins, ignored)
print(s.shape)

mu, sigma = 60, 10 # mean and standard deviation
x = np.ceil(np.random.normal(mu, sigma, 1000))
y = np.ceil(np.random.normal(mu+10, sigma+5, 1000))

from scipy import stats
print(stats.describe(x))
print(stats.describe(y))

# Covariance
#numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)[source]
print(np.cov(x,y))
X = np.vstack((x,y))
print(np.cov(X))
print(np.cov(x, y))
print(np.cov(x))
print(np.var(X, axis=1, ddof=1))

# Correlation
print(np.corrcoef(x,y))

import scipy
x = [5.05, 6.75, 3.21, 2.66]
print(scipy.stats.stats.rankdata(x))
#[ 3. 4. 2. 1.]

y = [1.65, 26.5, -5.93, 7.96]
print(scipy.stats.stats.rankdata(y))
#[ 2. 4. 1. 3.]

z = [1.65, 2.64, 2.64, 6.95]
print(scipy.stats.stats.rankdata(z))
#[ 1. 2.5 2.5 4.]

#This functionality is built into the R language:-
"""
x <- c(5.05, 6.75, 3.21, 2.66)
rank(x)
[1] 3 4 2 1
> y <- c(1.65, 26.5, -5.93, 7.96)
> rank(y)
[1] 2 4 1 3
> z <- c(1.65, 2.64, 2.64, 6.95)
> rank(z)
[1] 1.0 2.5 2.5 4.0
"""

import scipy
x = [5.05, 6.75, 3.21, 2.66]
y = [1.65, 26.5, -5.93, 7.96]
z = [1.65, 2.64, 2.64, 6.95]
print(scipy.stats.stats.spearmanr(x, y)[0])
#0.4
print(scipy.stats.stats.spearmanr(x, z))
print(scipy.stats.stats.spearmanr(x, z)[0])
#-0.55


import rpy2 as rpy

# Hello World for rpy2
import rpy2.robjects as ro
if __name__ == "__main__": print(ro.r('paste0("1 + 1 = ", 1 + 1)'))
ro.r('x = c(5.05, 6.75, 3.21, 2.66)')
ro.r('y = c(1.65, 26.5, -5.93, 7.96)')
#print(ro.r.cor(x,y, method="spearman"))
print(ro.r('cor(c(1,2,3),c(4,3,6))'))
print(ro.r('cor(x,y)'))

#0.4

print(ro.r('cor(x, y, method="spearman")'))

"""
This could be done in R as follows:

> x <- c(5.05, 6.75, 3.21, 2.66)
> y <- c(1.65, 26.5, -5.93, 7.96)
> z <- c(1.65, 2.64, 2.64, 6.95)
> cor(x, y, method="spearman")
[1] 0.4

> cor(x, z, method="spearman")
[1] -0.6324555
"""

import scipy
x = [5.05, 6.75, 3.21, 2.66]
y = [1.65, 26.5, -5.93, 7.96]
z = [1.65, 2.64, 2.64, 6.95]
print(scipy.stats.stats.kendalltau(x, y)[0])
#0.333333333333

print(scipy.stats.stats.kendalltau(x, z)[0])
#-0.547722557505

