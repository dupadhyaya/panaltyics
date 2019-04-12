# TS Plot
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('dailymintemp.csv',header=0)
series.plot()
pyplot.show()
