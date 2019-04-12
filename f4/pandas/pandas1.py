# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:04:24 2017 by Dhiraj Upadhyaya
"""
# Pandas
import pandas
pandas.__version__
#0.20.2

#Introduction
import numpy as np
import pandas as pd

#Panda Series Object
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data.index
data[1]
data[1:3]
data[0]

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a','b','c','d'])
data
data['b'], data[1]

#series as specialised Dictonary
population_dict = {'Delhi':25000, 'Noida':10000, 'Gurgaon':15000,'Faridabad':7500,'Ghaziabad':8000}
population_dict
population = pd.Series(population_dict)
population

population['Noida']

population['Delhi':'Gurgaon']

#Series Object
pd.Series([2,4,6])
pd.Series(5, index=[100,200,300])
pd.Series({1:'a',2:'b',3:'c'})
pd.Series({1:'a',2:'b',3:'c'}, index=[3,2])

#Data Frame Object
population_dict = {'Delhi':25000, 'Noida':10000, 'Gurgaon':15000,'Faridabad':7500,'Ghaziabad':8000}
population = pd.Series(population_dict)
population
area_dict = {'Delhi':100, 'Noida':30, 'Gurgaon':40,'Faridabad':25,'Ghaziabad':20}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'Population':population,'Area':area})
states = pd.DataFrame({population, area})
states
states.index
states.values
states.columns

#DF as spl Dictionary
states['Area']
states[:]
states[1::2]

#Constructing DF Objects
#from single series object
pd.DataFrame(population, columns=['Population'])
#from list for dict
data = [{'a':i,'b':i+1} for i in range(3)]
pd.DataFrame(data)

pd.DataFrame([{'a':1,'b':2}, {'b':3, 'c':4}])
#missing values with NaN
#fm dict of Series objects
population, area
pd.DataFrame({'Population':population, 'Area':area})

#From 2 dim Numpy
pd.DataFrame(np.random.rand(3,2), columns=['foo','bar'], index=['a','b','c'])

#from Numpy structured Array
A=np.zeros(3, dtype=[('A','i8'), ('B','f8')])
A
pd.DataFrame(A)


#Pandas Index Object
ind = pd.Index([2,3,5,7,11])
ind

#index as imutable array
ind[1]
ind[::2]
ind.size, ind.shape, ind.ndim, ind.dtype
ind[1]=0  #imutable
#index as ordered set
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])
indA, indB
indA & indB
indA | indB
indA ^ indB

#Data Indexing & Selection
#Data Selection in Series

import pandas as pd
data = pd.Series([0.25, 0.25,0.75, 1.0], index=['a','b','c','d'])
data
data['b']
'a' in data
data.keys()
list(data.items())
data['e'] = 0.25
data
data['e'] = 1.25
data
#Series as 1 Dim array
data['a':'c']
data[0:2]
data[(data > 0.3) & (data < 0.8)]
data[['a','e']]

#Indexer loc, iloc, ix
data = pd.Series(['a','b','c'], index=[1,3,5])
data
data[1]
data[1:3]
data.loc[1]
data.loc[1:3]
data.iloc[1]
data.iloc[1:3]

#data selection in DF
#DF as dict

pop = pd.Series({'Delhi':25000, 'Noida':10000, 'Gurgaon':15000,'Faridabad':7500,'Ghaziabad':8000})
pop
area  = pd.Series({'Delhi':100, 'Noida':30, 'Gurgaon':40,'Faridabad':25,'Ghaziabad':20})
area
data = pd.DataFrame({'area':area, 'pop':pop})
data
data['area']
data['pop']
data.area
data.area is data['area']
data.pop is data['pop']  #pop is a method
#z= data['pop'] #z #z in data['pop']

#DF as 2 dim array
data.values
data.T  #transpose
data.values[0]
data['area']

data.iloc[:3,:2]
data.loc[:'Faridabad', :'pop']
data.ix[:3, :'pop']

data['density'] = (data['area'] * 100/data['pop'])
data
data.loc[data.density > .3, ['pop','density']]
data.iloc[0,2] = .9
data

#Addl Conventions
data['Delhi':'Gurgaon']
data[0:4]
data[data.density > .3]

#Operating on Data in Pandas
