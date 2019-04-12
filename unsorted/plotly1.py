# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:45:42 2017 by Dhiraj Upadhyaya
"""
#Plotly
import plotly
print(plotly.__version__)  # version >1.9.4 required
from plotly.graph_objs import Scatter, Layout
plotly.offline.plot({ "data": [Scatter(x=[1, 2, 3, 4], y=[4, 1, 3, 7])], "layout": Layout( title="hello world")})

#Working

import plotly
help(plotly.offline.plot)


#Host into ur account
import plotly.plotly as py      # Every function in this module will communicate with an external plotly server
                      # use `py.iplot` inside the ipython notebook
py.plot({ "data": [{ "x": [1, 2, 3], "y": [4, 2, 5] }], "layout": {"title": "hello world"}}, filename ='helloworld.html', sharing='public')



import plotly.plotly as py
import plotly.graph_objs as go

labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500,2500,1053,500]
colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value',  textfont= dict(size=20), marker=dict(colors=colors, line=dict(color='#000000', width=2)))

py.iplot([trace], filename='styled_pie_chart')
