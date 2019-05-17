# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:00:33 2019

@author: du
"""
#links
#https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07

#pandas
import pandas
pandas.__version__
#0.23.0
#%%

#importing
import pandas as pd
#pd.<TAB>

pd?

import pandas as pd
flight_data = pd.read_csv(r"E:/pywork/pyProjects/panaltyics/data/nycflights13.csv")

flight_data.head()
flight_data.shape
flight_data.columns
