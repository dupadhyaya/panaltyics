# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:07:26 2019

@author: du
"""
#https://github.com/kieferk/dfply

from dfply import *
import pandas as pd


from pydataset import data
titanic = data('titanic')
titanic

#data >> select(X.first_col, X.second_col, X.third_col) >> drop(X.third_col) >> head(3))
type(titanic)
titanic.columns
titanic.index
titanic['class']
titanic.head()
titanic.tail()
titanic >> select(titanic.age, titanic.sex)
titanic >> drop(titanic.age, titanic.sex) >> head(3)
titanic >> select(~ titanic.age)  #notworking
#subset/filter
titanic >>  mask(titanic.sex=='man')
#sort
titanic >> arrange(titanic.sex, titanic.survived)
titanic >> arrange(titanic.sex, titanic.survived, ascending=False)

titanic >> group_by(titanic.sex) >>  summarize(count_sex = titanic.sex.count())

#----


#diamonds
#https://github.com/kieferk/dfply

from dfply import *
diamonds >> head(3)
diamonds >> select(X.carat, X.cut) >> head(3)
diamonds >> select(1, X.price, ['x', 'y']) >> head(2)
diamonds >> drop(1, X.price, ['x', 'y']) >> head(2)
diamonds >> select(~X.carat, ~X.color, ~X.clarity) >> head(2)
diamonds >> select(~[X.carat, X.color, X.clarity]) >> head(2)
#TypeError: bad operand type for unary ~: 'list'

diamonds >> group_by('cut') >> row_slice(5)
diamonds >> sample(frac=0.0001, replace=False)
diamonds >> sample(n=3, replace=True)
diamonds >> distinct(X.color)
diamonds >> mask(X.cut == 'Ideal') >> head(4)
diamonds >> mask(X.cut == 'Ideal', X.color == 'E', X.table < 55, X.price < 500)
diamonds >> filter_by(X.cut == 'Ideal', X.color == 'E', X.table < 55, X.price < 500)
(diamonds  >> filter_by(X.cut == 'Ideal', X.color == 'E', X.table < 55, X.price < 500) >> pull('carat'))
diamonds >> mutate(x_plus_y=X.x + X.y) >> select(columns_from('x')) >> head(3)
diamonds >> mutate(x_plus_y=X.x + X.y, y_div_z=(X.y / X.z)) >> select(columns_from('x')) >> head(3)
diamonds >> transmute(x_plus_y=X.x + X.y, y_div_z=(X.y / X.z)) >> head(3)


#group
(diamonds >> group_by(X.cut) >> mutate(price_lead =lead(X.price), price_lag=lag(X.price)) >> head(2) >> select(X.cut, X.price, X.price_lead, X.price_lag))
diamonds >> arrange(X.table, ascending=False) >> head(5)

(diamonds >> group_by(X.cut) >> arrange(X.price) >> head(3) >> ungroup() >> mask(X.carat < 0.23))

diamonds >> rename(CUT=X.cut, COLOR='color') >> head(2)

diamonds >> gather('variable', 'value', ['price', 'depth','x','y','z']) >> head(5)
diamonds >> gather('variable', 'value') >> head(5)

elongated = diamonds >> gather('variable', 'value', add_id=True)
elongated >> head(5)
widened = elongated >> spread(X.variable, X.value)
widened >> head(5)
widened.dtypes
widened = elongated >> spread(X.variable, X.value, convert=True)
widened.dtypes

d >> separate(X.a, ['col1', 'col2'], remove=True, convert=True,        extra='drop', fill='right')
