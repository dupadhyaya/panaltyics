# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:45:12 2017 by Dhiraj Upadhyaya
numpy2
"""

import numpy
#array([1,2,3])  # array not defd

#print(numpy.array([1,2,3]))
#print(type(numpy.array([1,2,3])))

import numpy as np
#print(np.array([1,2,3]))  # same short way

from numpy import array
#print(array([1,2,3]))  # 3rd way to create a array

"""
fam = ['liz',1.73,'emma', 1.68, 'mom', 1.71, 'dad', 1.81 ]
print(fam, ' No of elements ', len(fam), sep=' ')
fam_ext = fam + ['me', 1.79]
print(fam_ext, ' length of elements', len(fam_ext), sep=' ')
np_fam = array(fam_ext)
print(np_fam)
"""

# Elementwise calcualations
sister = 'liz'
height = 1.73
np_height = np.array(height)
print(np_height)
height = [1.73, 1.68, 1.71, 1.81, 1.79 ]
weight = [65.5, 59.2, 63.6, 88.4, 68.7 ]
print(type(height), height, sep=' ')
height2 = array([1.73, 1.68, 1.71, 1.81, 1.79 ])
weight2 = array([65.5, 59.2, 63.6, 88.4, 68.7 ])
print(type(height2), height2, sep=' ')
print( height2, weight2, sep=' ')
bmi = weight2/ height2
print(np.around(bmi,3)) # or numpy.around()
print(np.round(sum(bmi),2))  # round also same

#bmi2 = height/weight ** 2   # not supported
bmi2b = weight2/ height2 **2 
print(bmi2b)

nparr2 = np.array([1, 'abc', 2.3, True])
print(nparr2, type(nparr2), sep=' ')  # convert all to text, has only data type
print(bmi)
print(bmi[2])
print(bmi > 38)
print(bmi[bmi > 38])
print(type(nparr2))


