# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:50:52 2017 by Dhiraj Upadhyaya
Plot
"""

"""
import matplotlib.pyplot as plt
vals = [3,2,5,0,1]
plt.plot(vals)

#plt.show()

"""


votes = {
    'White Icicle': 65,
    'Snow Belle': 63,
    'Champion': 76,
    'Cherry Belle': 58,
    'Daikon': 63,
    'French Breakfast': 72,
    'Bunny Tail': 72,
    'Sicily Giant': 57,
    'Red King': 56,
    'Plum Purple': 56,
    'April Cross': 72
}



import matplotlib.pyplot as plt
import numpy as np

names = []
votes = []
# Split the dictionary of name:votes into two lists, one for names and one for vote count
for radish in counts:
    names.append(radish)
    votes.append(counts[radish])

# The X axis can just be numbered 0,1,2,3...
x = np.arange(len(counts))

plt.bar(x, votes)
plt.xticks(x + 0.5, names, rotation=90)
