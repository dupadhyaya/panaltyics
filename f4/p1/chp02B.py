# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:44:18 2017 by Dhiraj Upadhyaya
Chp-2B Movie Lens
"""
import pandas as pd
unames = [ 'user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('./data/users.dat', sep='::', header=None, names=unames, engine='python')
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('./data/ratings.dat', sep='::', header=None, names=rnames, engine='python')
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('./data/movies.dat', sep='::', header=None, names = mnames, engine='python')

print(users[:5])
print(ratings.tail())
print(movies.head())
print(ratings)

data = pd.merge(pd.merge(ratings, users), movies)
print(data.head())
print(data.ix[0]) #data.iloc[0])
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print(mean_ratings[:5])
