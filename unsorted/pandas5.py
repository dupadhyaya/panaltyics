# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:59:48 2017 by Dhiraj Upadhyaya
"""
#pandas5
!curl 'https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv'
import pandas as pd
pop = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv')
areas = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv')
abbrevs = pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv')
pop.head()
areas.head()
abbrevs.head()
merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
merged
merged.head()
merged.isnull().any()
merged[merged['population'].isnull()].head()
merged.[merged['state'].isnull(), 'state/region'].unique()
merged.loc[merged['state/region']=='PR','state'] = 'Puerto Rico'
merged.loc[merged['state/region']=='USA','state'] = 'United States'
merged.isnull().any()
#Pg156

final = pd.merge(merged, areas, on='state', how='left')
final.head()

final.isnull().any()
#nulls in area
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace=True)
final.head()

data2010 = final.query("year ==  2010 & ages == 'total'")
data2010.head()['area (sq. mi)']

type(data2010)
data2010.iloc[1:5:, 4:7]
data2010.set_index('state', inplace=True)
density = data2010['population']/data2010['area (sq. mi)']
density
density.sort_values(ascending=False, inplace=True)
density
density.tail()

#Aggregation and Grouping
import seaborn as sns
planets = sns.load_dataset('planets') #net was reqd ?
planets.shape
planets.head()
type(planets)
planets.describe()

#aggregation
rng = np.random.RandomState(42)
rng
ser = pd.Series(rng.rand(5))
ser
ser.sum()

df = pd.DataFrame({'A':rng.rand(5),'B':rng.rand(5)})
df
df.mean()
df.mean(axis=1)
df.mean(axis=0)

df.describe()
df.dropna.describe()
planets.describe()

#Groupby: Split, Apply, Combine
df= pd.DataFrame({'key':['A','B','C','A','B','C'], 'data':range(6)}, columns=['data','key'])
df
df.groupby('key')
df.groupby('key').aggregate('mean')
df.groupby('key').sum()
df.groupby('key').mean()

planets.groupby('method')
planets.columns
planets.groupby('method')['orbital_period']
planets.groupby('method')['orbital_period'].sum()

#Iteration over groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))
#error
#planets.groupby('method').groups

planets.groupby('method')['year'].describe().unstack()

#aggregate
rng = np.random.RandomState(0)
df = pd.DataFrame({'key':['A','B','C','A','B','C'], 'data1':range(6),'data2':rng.randint(0,10,6)}, columns = ['key','data1','data2'])
df

df.groupby('key').aggregate('min')
df.groupby('key').aggregate([min,np.median, max])
df.groupby('key').aggregate({'data1':'min', 'data2':'max'})

#filtering
def filter_func(x):
    return x['data2'].std() > 4
df.groupby('key').std()
df.groupby('key').filter(filter_func())  #not working

#transformation
df.groupby('key').transform(lambda x: x - x.mean())

#apply
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x
#norm_by_data2(df)
df.groupby('key').apply(norm_by_data2))
df.groupby('key').apply(max)


#Split
L = [0,1,0,1,2,0]
df
df.groupby(L).sum()
df.groupby(df['key']).sum()
df2 = df.set_index('key')
df2
mapping = {'A':'Delhi','B':'Noida','C':'Delhi'}
mapping
df2.groupby(mapping).sum()

df2.groupby(str.lower).mean()
df2.groupby([str.lower, mapping]).mean()


#Grouping Example
decade = 10 * (planets['year'] // 10)
decade
decade = decade.astype(str) + 's'
decade
planets.groupby(['method',decade])['number'].sum().unstack().fillna(0)

