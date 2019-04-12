# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:20:30 2018
Dhiraj
@author: du
"""
import numpy as np
import pandas as pd

#%%
import random
for x in range(11):
  print(random.randint(50,100))

#%% List comprehension
markL = [random.randint(50,100) for i in range(11) ]  
markL

#%%
from random import randrange, uniform

irand = randrange(0, 10)
irand
# uniform gives you a floating-point value
frand = uniform(0, 10)
frand


#%%
from random import randint
x=[randint(0,9) for p in range(0,9)]
x

#%%


rng = np.random.RandomState(10)
marks1 = rng.randint(50,100,11)
marks1
marks2 = pd.Series(rng.rand(11))
marks2
list(marks2)
marks3 = pd.Series(rng.randint(50,100,11))
marks3


#%%
from secrets import randbelow
print(randbelow(10))


#%%
import random
nums = [x for x in range(10)]
nums
random.shuffle(nums)
nums

#%%

import random
values = list(range(10))
values
random.choice(values)

#choice also works for one item from a not-continuous sample:

values = [1, 2, 3, 5, 7, 10]
random.choice(values)

#If you need it "cryptographically strong" 
import secrets
values = list(range(10))
secrets.choice(values)

#%% 1 to 10, 20 nos
import numpy as np   
np.random.randint(10, size=(1, 20))

import numpy as np
print(np.random.randint(0,10))


#%%
import random
N = 5
[random.randint(0, 9) for _ in range(N)]
# [9, 7, 0, 7, 3]

[random.choice(range(10)) for _ in range(N)]


#%%
#random.sample returns k unique selections from a population (without replacement):2
N=5
random.sample(range(10), k=N)  #without replacement

#random.choices returns k selections from a population (with replacement):

random.choices(range(10), k=N)
# [3, 2, 0, 8, 2]

#%%
import random
random.randrange(10)

random.randrange(10)

#To get a list of ten samples:

[random.randrange(10) for x in range(10)]

#%%https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
n={} 
for i in range(10):
    n[i]=i

for p in range(10):
    print(n.popitem()[1])


#%%  Seed
"""
numpy.random.seed() to set the seed. 
Using random.seed() will not set the seed for random numbers generated from numpy.random. """

np.random.seed(10)
marks1 = np.random.randint(50,100,10)
marks1
np.random.seed(10)
marks2 = np.random.randint(50,100,10)
marks2


import random
random.seed(5), random.randint(1,10)
random.seed(5), random.randint(1,10)
random.randint(1,10)


#%% System Dependent
import sys
import random

key_num = random.SystemRandom()
print(key_num.random()) #produces a number between 0 and 1
print(key_num.randint(0, sys.maxsize)) # produces a integer between 0 and the highest allowed by the OS.
