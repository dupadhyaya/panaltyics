#Time Series Data Visualization with Python
#http://machinelearningmastery.com/time-series-data-visualization-with-python/
#Minimum Daily Temperatures Dataset
#https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990#!ds=2324&display=line
# Delete ? and footer by editing the file
'''
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('dailymintemp.csv',sep=";", header=0)

print(series[:1])
print(series.head())
'''
#http://machinelearningmastery.com/load-machine-learning-data-python/
import csv
import numpy
filename = 'pima-indians-diabetes-data.csv'
raw_data = open('pima-indians-diabetes-data.csv')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape) # No of records and attributes