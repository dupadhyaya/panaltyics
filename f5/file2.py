# Load CSV from URL using NumPy
'''
import numpy
import urllib
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = numpy.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
'''

import urllib.request
local_filename, headers = urllib.request.urlretrieve('http://python.org/')
html = open(local_filename)

#workfile = 'dailymintemp.csv'
#f = open('workfile', 'w')
#f.readline()
'''
import csv
with open('dailymintemp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
print(writer)
'''

import csv
with open('dailymintemp.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)