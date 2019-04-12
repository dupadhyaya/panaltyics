# -*- coding: utf-8 -*-
"""
Tue Feb 13 12:49:02 2018: Dhiraj
"""

#aggregations

marksA = np.random.randint(50,100,11)
marksA

marksA.sum()
courseL = ['bsc', 'pg', 'msc']
import random
courseL1 = random.choice(courseL,10)
courseL1  #single item
courseRL =[]
for i in range(11):
    courseRL.append(random.choice(courseL))
courseRL

#%%

import random
 
nameL = ['meena','apoorva','kastav','shubam', 'goldie','hitesh','shruti','vijay','achal','lalit','varun']
 
rand_item = nameL[random.randrange(len(nameL))]
rand_item


rand_items = [nameL[ random.randrange( len(nameL))] for item in range(4)]
rand_items
rand_item = random.choice(items)

rand_items = random.sample(items, n)

"""sets are not indexable, meaning set([1, 2, 3])[0] produces an error. Therefore random.choice does not support sets, however random.sample does."""


from random import choice, sample
# Convert the set to a list
choice(list(set(nameL)))
# random.sample(), selecting 1 random element
sample(set(nameL), 1)
sample(set(nameL), 1)[0]


#%%
my_list = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
my_set = set(my_list)
my_set
my_list = list(my_set) # No duplicates
my_list
my_elem = random.choice(my_list)
my_elem
another_elem = random.choice(list(set([1, 1, 1])))
another_elem
