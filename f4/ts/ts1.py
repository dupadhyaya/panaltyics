# -*- coding: utf-8 -*-
"""
Mon Feb 12 11:10:58 2018: Dhiraj
"""
#build a date
from datetime import datetime
datetime(year=2015, month=7, day=4)

import datetime
today= datetime.date.today()
today
now = datetime.datetime.now()
now

mydate ="1967-06-23"
mydate1 = datetime.datetime.strptime(mydate, "%Y-%m-%d").date()
mydate1
diff = (today - mydate1).days
diff

# parse dates
from dateutil import parser
date  = parser.parse("4th of July, 2015")
date
type(date)

#use it different ways : print day of the date
date.strftime('%A')

#pytz - contains tools for working with timezones
#some weakeness in datetime format -> datetime64 in numpy

# array of times
import numpy as np
date = np.array('2018-01-01', dtype=np.datetime64)
date

date + 4  # nothing
date + range(4)  # next 4 days

date + np.arange(12)
# uniform data types in np - faster
#timedelta64
#built on fundamental

np.datetime64('2017-02-12') 
np.datetime64('2017-02-12 09:00') 
np.datetime64('2017-02-12 09:00', 'ns') 
np.datetime64('2017-02-12 09:00:59.50', 'ns') 

#Description of date and time codes

#date time in Pandas
import pandas as pd
date = pd.to_datetime("02nd Feb 2018")
date
date.strftime('%A')

date + pd.to_timedelta(np.arange(12), 'D')

# Indexing by time
index = pd.DatetimeIndex(('2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04'))
index
data = pd.Series([0,1,2,3], index=index)
data
data['2018']

#month ?
