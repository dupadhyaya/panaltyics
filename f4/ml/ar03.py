# -*- coding: utf-8 -*-
"""
Tue Feb 27 10:53:44 2018: Dhiraj
"""
import numpy as np
from itertools import combinations, groupby
from collections import Counter

# Sample data
orders = np.array([[1,'apple'], [1,'egg'], [1,'milk'], [2,'egg'], [2,'milk']], dtype=object)

# Generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    # For each order, generate a list of items in that order
    for order_id, order_object in groupby(orders, lambda x: x[0]):
        item_list = [item[1] for item in order_object]      
#For each item list, generate item pairs, one at a time
        for item_pair in combinations(item_list, 2):
            yield item_pair                                      


# Counter iterates through the item pairs returned by our generator and keeps a tally of their occurrence
Counter(get_item_pairs(orders))

import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from IPython.display import display

# Function that returns the size of an object in MB
def size(obj):
    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))

orders = pd.read_csv('../input/order_products__prior.csv')
print('orders -- dimensions: {0};   size: {1}'.format(orders.shape, size(orders)))
display(orders.head())

# Convert from DataFrame to a Series, with order_id as index and item_id as value
orders = orders.set_index('order_id')['product_id'].rename('item_id')
display(orders.head(10))
type(orders)

print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'.format(orders.shape, size(orders), len(orders.index.unique()), len(orders.value_counts())))