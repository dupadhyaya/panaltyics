# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:00:33 2019

@author: du
"""
#links
#https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07

#pandas
import pandas
pandas.__version__
#0.23.0
#%%

#importing
import pandas as pd
#pd.<TAB>
pd?

import pandas as pd
#import from local PC
flight_data = pd.read_csv(r"E:/pywork/pydata/nycflights13.csv")
#import from URL
url1="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
countries=pd.read_csv(url1)
countries.head()
url2 = "https://raw.githubusercontent.com/dupadhyaya/hheanalytics/master/data/mtcars.csv"
mtcars = pd.read_csv(url2)
mtcars.head()

#use any one

data = mtcars
#data = flight_data
data.head()
data.shape
data.columns
data.values
data.T
data['mpg']
data.iloc[:3, :2] #index
data.loc[:4, :'hp'] #name
data.ix[:5, :'wt']  #mixed
#filter
data.loc[data.mpg > 25,]
data.loc[data.mpg > 25,['mpg','hp']]
data[0:4]
data[data.cyl > 4]
data['mpg':'hp']

#index
data2= data.set_index('carname')   #create rowname
data2
#combining datasets
pd.merge(mtcars, data)
pd.merge(mtcars, data, on='carname')
pd.merge(mtcars, data, left_on='carname', right_on='carname')

pd.merge(mtcars, data, left_on='carname', right_on='carname').drop('carname', axis=1)



#%%
#simple aggregations
data2.shape
data2.describe
data2.mean()
data2.sum()
data2.mean(axis='columns')
data2.mean(axis='rows')
data2.dropna().describe()
data2.describe()
data2.count()
data2.mpg.min()
data2.min(), data2.min(axis='rows')
data2.max(), data2.max(axis='columns') #default is columns wise ie axis=rows
data2.std()
data2.var()
data2.mad()
data2.prod()
data2.sum()
data2.mpg.sum()
data2.mpg.mean()

#Groupby
data2.groupby('am')  #nothing will happen
data2.groupby('am').sum()
data2.groupby(['am','gear']).sum()
data2.groupby('am').sum()['mpg']
data2.groupby('am')['mpg'].sum()

#how many rows and columns in each group
for (am, group) in data2.groupby('am'):
    print(am, group.shape)
    
#summary groupwise 
data2.groupby('am')['wt'].describe().unstack()
data2.groupby(['am','cyl'])['wt'].describe()

data2.groupby('am')['mpg','carb'].aggregate(['min',np.median,'max']).unstack()
data2.groupby('am').aggregate({'mpg':mean, 'wt':max})
data2.groupby('am').aggregate({'mpg':'mean', 'wt':'max', 'hp':median})

data2.groupby(['am','cyl']).size()
data2.groupby(['am','cyl']).size().reset_index(name='counts')
#
(data2.groupby(['am', 'cyl']).agg({'mpg': ['mean', 'count'], 'gear': ['median', 'min', 'count'] }))


#make a group
grp = data.groupby(['am',  'cyl']) 
grp.max() 
grp.mean() 
grp.describe() 
grp.count().reset_index()  #level of headings

#transform
data2.groupby(['am',  'cyl']).transform(lambda x: x- x.mean())
data3 =data2
data3["ratio"] = ""
data3=data3.assign(C="",D=np.nan)
data3.columns
data3.reindex(columns=list('ABCD'))
data3[['mpg','ratio','C','D']]
def nfunc(x):
    x['D'] /= x['mpg']/x['wt']
    return(x)

data3.groupby(['am',  'cyl']).apply(nfunc)

#%%
#adding columns
data3["ratio"] = ""
data3=data3.assign(C="",D=np.nan)
v\data3.reindex(columns=list('ABCD'))


#%% specifying split key
L=[4,6,8]
data2.groupby('gear').sum()

datagear = data2.set_index('gear')
mapping={3:'G3', 4:'G4', 5:'G5'}
datagear.groupby(mapping).sum()

#%%
#pivot table
data2.head()
data2.groupby('gear')['mpg'].mean()
data2.groupby(['gear','am'])['mpg'].mean()
data2.groupby(['gear','am'])['mpg'].aggregate('mean')
data2.groupby(['gear','am'])['mpg'].aggregate('mean').unstack()

data2.pivot_table('mpg', index='am', columns='gear')

#multilevel
mileage = pd.cut(data2['mpg'], [0,15,20,25,30,35])
mileage
data2.pivot_table('vs',mileage)
data2.pivot_table('hp',['gear',mileage], aggfunc='sum')
data2.pivot_table(values='hp', columns='am',index=['gear',mileage], aggfunc=np.mean)
#index='leftColn', columns='topCol', values='ValueCol', aggfunc==np.mean)/sum/first/last/np.std, aggfunc=', '.join, fill_value='-')
#aggfunc={'Age': np.mean, 'Random': [np.mean, np.std]})]
data2.pivot_table(columns='cyl',index='gear', values='am', aggfunc='count', margins=True)
#plot
data2.pivot_table(columns='cyl',index='gear', values='am', aggfunc='count').plot()

#eval
data2.eval('newMPG = mpg * 3')
#inplace=True will create new column