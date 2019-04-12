# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 06:57:05 2017 by Dhiraj Upadhyaya
https://www.marsja.se/pandas-python-descriptive-statistics/
"""
import numpy as np
from pandas import DataFrame as df
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean
import pandas as pd
N = 20
P = ["noise","quiet"]
Q = [1,2,3]
 
values = [[998,511], [1119,620], [1300,790]]
 
mus = np.concatenate([np.repeat(value, N) for value in values])
 
data = df(data = {'id': [subid for subid in range(N)]*(len(P)*len(Q)),'iv1': np.concatenate([ np.array([p]*N) for p in P]*len(Q)),'iv2': np.concatenate( [np.array([q]*(N*len(P))) for q in Q]),'rt': np.random.normal( mus, scale=112.0, size=N*len(P)*len(Q))})

data.describe()
data
grouped_data = data.groupby(['iv1', 'iv2'])
grouped_data['rt'].describe().unstack()
grouped_data['rt'].mean().reset_index()
grouped_data['rt'].aggregate(np.mean).reset_index()
trimmed_mean = grouped_data['rt'].apply(trim_mean, .1)
trimmed_mean.reset_index()

grouped_data['rt'].median().reset_index()
grouped_data['rt'].aggregate(np.median).reset_index()
grouped_data['rt'].median()
grouped_data['rt'].apply(mode, axis=None).reset_index()
#here is a method (i.e., pandas.DataFrame.mode()) for getting the mode for a DataFrame object. However, it cannot be used on the grouped data so  use mode from SciPy:


descr = grouped_data['rt'].aggregate([np.median, np.std, np.mean]).reset_index()
descr
descr['trimmed_mean'] = pd.Series(trimmed_mean.values, index=descr.index)
descr

grouped_data['rt'].std().reset_index()

grouped_data['rt'].quantile([.25, .5, .75]).unstack().reset_index()

grouped_data['rt'].var().reset_index()
