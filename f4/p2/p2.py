# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 06:58:44 2017 by Dhiraj Upadhyaya
"""

from numpy import arange, cos, linspace, pi, sin, random
from scipy.interpolate import splprep, splev
t = linspace(0, 1.75* pi, 100)
x = sin(t)
y = cos(t)
z = t

print(z)
