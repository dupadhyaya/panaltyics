# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:24:34 2018
Dhiraj
@author: du
"""
#http://members.cbio.mines-paristech.fr/~nvaroquaux/formations/scipy-lecture-notes/intro/numpy/operations.html
import numpy as np
from scipy import misc
lena = misc.face()
import pylab as plt
lena = misc.face()
plt.imshow(lena)
plt.imshow(lena, cmap=plt.cm.gray)
crop_lena = lena[30:-30,30:-30]
crop_lena
y, x = np.ogrid[0:512,0:512] # x and y indices of pixels
y.shape, x.shape
centerx, centery = (256, 256) # center of the image
mask = ((y - centery)**2 + (x - centerx)**2) > 230**2 # circle
#then we assign the value 0 to the pixels of the image corresponding to the mask. The syntax is extremely simple and intuitive:

lena[mask] = 0
plt.imshow(lena)
