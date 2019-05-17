# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:41:40 2019

@author: du
"""

#file import and export

import os
print(os.getcwd())
print(os.listdir(os.getcwd()))
os.chdir('..')

#change working directory
#E:\pyWork\pyProjects\panaltyics
path="E:/pyWork/pyProjects/panaltyics"
os.chdir(path)
os.getcwd()

import pandas as pd
flight_data = pd.read_csv(r"E:/pywork/pyProjects/panaltyics/data/nycflights13.csv")

flight_data.head()
