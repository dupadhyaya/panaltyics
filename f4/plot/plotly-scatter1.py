# -*- coding: utf-8 -*-
"""
Mon Mar 12 11:29:00 2018: Dhiraj
"""
import plotly
plotly.__version__

import plotly.plotly as py
import plotly.graph_objs as go

#create random data with numpy
import numpy as np
N=1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

#create a trace go.Scatter?
trace = go.Scatter(x = random_x, y=random_y, mode='markers')
trace[1:5,1:5]
data = [trace]
py.iplot(data, filename='basic-scatter')
