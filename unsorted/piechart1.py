# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:48:18 2017 by Dhiraj Upadhyaya
"""

#Pie


#Method1
import matplotlib.pyplot as plt
 
# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#Method2
import matplotlib.pyplot as plt
 
labels = ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake']
sizes = [38.4, 40.6, 20.7, 10.3]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()



#Method2
import plotly
plotly.tools.set_credentials_file(username='dup1966 ', api_key='0zZ5FLEa1sM0EfqsMK9R')

import plotly as py1
import plotly.plotly as py
import plotly.graph_objs as go

labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500,2500,1053,500]

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='basic_pie_chart')
