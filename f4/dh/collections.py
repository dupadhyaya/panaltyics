# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 07:00:01 2017 by Dhiraj Upadhyaya
"""
from collections import Counter
# Tally occurrences of words in a list

for i in range(10): print('Hi', end=' ')

cnt = Counter()
for word in (['red', 'blue', 'red', 'green', 'blue', 'blue']): 
    cnt[word] += 1
cnt
#Counter({'blue': 3, 'red': 2, 'green': 1})

# Find the ten most common words in a file
import re
words = re.findall(r'\w+', open('ftemps.txt').read().lower())
Counter(words).most_common(10)


for x in range(0,3):
    print("We're on time %d" % (x))
    print('Dhiraj')

string = "Hello World"
for x in string:
    print(x , end=' ')
    print(x, end=' ')
