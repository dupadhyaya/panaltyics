# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:22:23 2017 by Dhiraj Upadhyaya
"""
#Vectorised String Operations
#pandas7
import numpy as np
x = np.array([2,3,5,7,11,13])
x
x*2
data = ['achal', 'apporva','goldie', 'hitesh']
data
[s.capitalize() for s in data]

data = ['achal', 'apporva',None, 'HitesH']
[s.capitalize() for s in data]
#error

import pandas as pd
names = pd.Series(data)
names
names.str.capitalize()

#Sting methdds
monte = pd.Series(['Achal Kumar','Apoorva Karn','Goldie Sahni', 'Hitesh Mann', 'Kaustav Chatterjee', 'Lalit Sahni', 'Meena Srisha Valavala', 'Shruti Sinha', 'Shubham Pujan', 'Vijay Pal Singh'])
monte
monte.str.lower()
monte.str.len()
monte.str.startswith('A')
monte.str.split()
monte.str.extract('([A-Za-z]+)')

monte.str.findall(r'^[^AEIOU].*[^aeiou]$')

monte.str.slice()

monte.str[0:3]
monte.str.split().str.get(-1)
monte

full_monte = pd.DataFrame({'name': monte, 'info':['B|C|D', 'B|D', 'A|B|C', 'A|C','B|C','A|B|C','A|B|C', 'A|C','B|C','A|B|C']})
full_monte
full_monte['info'].str.get_dummies('|')

#Example Recipe
#!curl -O http://openrecipes.s3.amazonaws.com/recipeitems-latest.json.gz
#!gzip recipeitems-latest.json.gz

recipies = pd.read_json('recipies.json')
recipies
recipies.shape
f = open('recipies.json')
f
line = f.readline()
pd.read_json(line).shape
recipies.shape
recipies.iloc[0]
recipies.columns
recipies.ingredients.str.len().describe()
recipies.name[np.argmax(recipies.ingredients.str.len())]
recipies['steps']
