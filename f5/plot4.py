#import matplotlib.pyplot as plt
from pandas import Series
import pandas as pd
from numpy.random import randn
ts = Series(randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()