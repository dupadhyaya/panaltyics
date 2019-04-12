# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:53:30 2017 by Dhiraj Upadhyaya
"""
#Pandas - Pg 128
import pandas as pd
import numpy as np

pop = pd.Series({'Delhi':25000, 'Noida':10000, 'Gurgaon':15000,'Faridabad':7500,'Ghaziabad':8000})
pop
area  = pd.Series({'Delhi':100, 'Noida':30, 'Gurgaon':40,'Faridabad':25,'Ghaziabad':20})
area
pop/area

index  = [('Delhi',2015),('Delhi',2016),('Noida',2015), ('Gurgaon',2015),('Gurgaon',2016),('Faridabad',2015),('Ghaziabad',2015)]
index
populations = [25000, 25700, 10000, 11000, 15000, 16000,7500]
len(populations)
pop = pd.Series(populations, index=index)
pop
pop[('Delhi',2015):('Gurgaon',2015)]

#Multi-index

index = pd.MultiIndex.from_tuples(index)
index
pop = pop.reindex(index)
pop

pop[:2016]
pop[:2015]
pop[:2017]

pop_df = pop.unstack()
pop_df
pop_df.stack()

pop_df = pd.DataFrame({'total':pop, 'under18':[10000, 11000, 5000, 3000, 4000, 5000, 2000]})
pop_df

f_u18 = pop_df['under18']/ pop_df['total']
f_u18
f_u18.unstack()

#Indexed Arrays
df = pd.DataFrame(np.random.rand(4,2), index=[['a','a','b','b'], [1,2,1,2]], columns=['data1','data2'])
df

data  = {('Delhi',2015):25000,('Delhi',2016):25700,('Noida',2015):10000, ('Gurgaon',2015):11000,('Gurgaon',2016):15000,('Faridabad',2015):16000,('Ghaziabad',2015):7500}
type(data)
data
pd.Series(data)

#Explicity Multiindex Constructors
pd.MultiIndex.from_arrays([['a','a','b','b'], [1,2,1,2]])
pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
pd.MultiIndex.from_product([['a','b'],[1,2]])
pd.MultiIndex(levels=[['a','b'],[1,2]], labels=[[0,0,1,1],[0,1,0,1]])

pop.index.names=['state','year']
pop

#Multi Index for columns
index = pd.MultiIndex.from_product([[2016,2017],[1,2]], names=['year','semester'])
columns = pd.MultiIndex.from_product( [['DS','Law','ANM'],['Sub1','Sub2']], names=['subject','type'])
data = np.round(np.random.randn(4,6),1)
data
data[:, ::2] *= 10
data
data += 37
data
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
health_data['DS']

#Indexing & Slicing a MultiIndex
pop
pop['Noida',2015]
pop['Delhi']
pop.loc['Delhi':'Noida']  #not sorted 
pop
pop[:, 2015]
pop[pop > 10000]
pop[['Noida', 'Gurgaon']] # not sorted

#Multiply indexed DF Pg 136
health_data
health_data['DS','Sub1']
health_data.iloc[:2,:2]

health_data.loc[:, ('Law','Sub2')]
#slice
idx = pd.IndexSlice
health_data.loc[idx[:,1:2], ]
#health_data.loc[idx[:,:1]]  #sort

#Rearranging Multi Indexes
index = pd.MultiIndex.from_product([['a','c','b'],[1,2]])
index
data = pd.Series(np.random.rand(6), index=index)
data
data.index.names = ['char','int']
data
data['a':'b']

data=data.sort_index()
data
data['a':'b']

#Stacking & Unstacking Indices
pop.unstack(level=0)
pop.unstack(level=1)
pop.unstack(level=1).stack()

#Index Setting & Resetting
pop_flat = pop.reset_index(name='population')
pop_flat
pop
pop_flat.set_index(['state','year'])


#Data Aggregation on Multi Indices
health_data
data_mean = health_data.mean(level='semester')
data_mean
health_data.mean(level='year')
health_data.mean(axis=1, level='type')
health_data.mean(axis=1)
health_data.mean(axis=0)
