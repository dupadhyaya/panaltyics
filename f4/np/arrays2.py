# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:47:51 2017 by Dhiraj Upadhyaya
"""

import numpy as np

np.random.seed(0)
x0 = np.random.randint(10); x0
x0[0]

np.random.seed(0)
x1 = np.random.randint(10, size=6); x1
x1[0:2]
x1[:-1]

#np.random.seed(0)
#2dim array
x2 = np.random.randint(10, size=(3,4))
x2
x2[-3,-4]
x2[:,0],x2[:,1],x2[:,-2],
x2[:2, :3]
x2[:3, ::2]
x2[0:1,:]
x2[::2,:]
x2.extend([10,11,12])

x2[:, np.newaxis]

x2a = nd.



# dim array
x3 = np.random.randint(10, size=(3,4,5))
x3  #3m x 4r x 5c
x3[0::]
x3[1::]
x3[2::]
x3[1:1]

x3.sum()
x3.sum(axis=0)
x3.sum(axis=1)
x3.sum(axis=2)
x3.sum(axis=none)



x5 = np.array([1,2,3,4,5,6])
x5.sum()
type(x5)

x5a = x5.reshape(2,3)
x5a.sum()
x5a.sum(axis=1)
x5a.sum(axis=0)


x5b= x5a.reshape(3,2)
x5b
x5.reshape(3,2)
x5b.sum()
x5b.sum(axis=0)
x5b.sum(axis=1)

#Add elements to array

x6= np.append(x5, [7,8,9])
x6
x6.sum()

x5a
x6a = np.append(x5a,[[7,8,9]], axis=0)
x6a
x6a.sum()

x5b
x6b = np.append(x5b,[[7],[8],[9]], axis=1)
x6b
x6b.sum()

np.resize(x6b, (1,9))
np.resize(x6b, (9,1))

