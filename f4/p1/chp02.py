# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:06:11 2017 by Dhiraj Upadhyaya
Chp-2 :Introductory Examples
"""
path = 'data/example.txt'
records1 = open(path).readline()
#print(records1) # first line

import json
records = [json.loads(line) for line in open(path)]
#print(records)  # print all records
#print(records[0])  # first record

#print(records[0]['tz'])  # Time zone of 1st record

#counting timezones
#time_zones = [rec['tz'] for rec in records]
#print(time_zones)  # error as not all have tz data

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
#print(time_zones[:10], sep='\n')
#some are empty  - remove them - different methods

"""
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

#get_counts()
#print(get_counts(time_zones), sep='\t', end=' ')  # working

#method2
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int) # values will intialise to 0
    for x in sequence:
        counts[x] += 1
    return counts
counts = get_counts2(time_zones)
#print(get_counts2(time_zones))
print('Count of TZ ', counts, sep='\t')
print("No of Time Zones ", len(time_zones))

print('Count of America/ NY is ', counts['America/New_York'])
print('Count of Asia/Dubai is ', counts['Asia/Dubai'])


#pg 20 Top 10 times and their counts
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]
print(' Top Counts ', top_counts(counts,10))
#view = reversed(sorted(elements))
view = reversed(sorted(top_counts(counts,10))) 
for element in view:
    print(element)

#using Python standard Library - collections - counter
from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(5))

"""
#Pandas ------
from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
#print(frame)

print(frame['tz'][:10])  # first 10 values
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
#print(tz_counts)

# To draw a plot, remove missing/empty/ NA values
clean_tz = frame['tz'].fillna('Missing')  # NA with Missing
clean_tz[clean_tz == ''] = 'Unknown' # empty strings with Unknown
tz_counts = clean_tz.value_counts()
tz_counts[:10]
print(tz_counts[:10])

# draw bar plot
tz_counts[:10].plot(kind='barh', rot=0)

print(frame['a'][1])
print(frame['a'][50])
print(frame['a'][51])
print(frame[0:2][51])
