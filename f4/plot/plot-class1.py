# -*- coding: utf-8 -*-
"""
Mon Mar 12 11:59:53 2018: Dhiraj
"""
import matplotlib.pyplot as plt
#Class practise
x = [1,2,3]
y = [2,4,1]
#run together
plt.plot(x,y)
#plt.plot(x,y, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()


# Two or more lines

x1 = [1,2,3]
y1 = [2,4,1]
x2 = [1,2,3]
y2 = [4,1,3]
#run together
plt.plot(x1,y1, label='Line1')
plt.plot(x2,y2, label='Line2')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Two Lines on the same plot')
plt.show()

# Bar Chart
x1, y1
tick_label= ['One', 'Two', 'Three']
plt.bar(x,y, tick_label= tick_label, width=0.8, color=['red','green'])
plt.xlabel('Group')
plt.ylabel('Values')
plt.title('Tile Bar Chart')
plt.show()

# Histogram
#freq  np.random.uniform 
marks = np.random.uniform(30,100,1000)
marks
np.all(marks >= 30)
np.all(marks < 100)
range = (20,100)
bins = 10
plt.hist(marks)
plt.hist(marks, bins, range, color='green', histtype='bar', rwidth=0.8)
plt.xlabel('Marks')
plt.ylabel('No of Students')
plt.title('Histogram of Marks of Students')

#count, bins, ignored = plt.hist(s, 15, normed=True)
#plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

# Scatter Plot
x,y
plt.scatter(x,y)
plt.scatter(x,y, label='stars', color='red', marker='*', s=30)
plt.legend()
plt.show()


# Pie Chart
x,y
y
activity = ['sleep','study','eat']
colors = ['red','green','yellow']
plt.pie(y)
plt.pie(y, labels=activity, colors = colors)
plt.pie(y, labels=activity, colors = colors, startangle=45, shadow=True, radius=2, explode=(0.5,0.0,0.0), autopct = '%1.1f%%')
#rotate start of pie by 90deg, explode offset each wedge, autopct - label format
plt.legend()
plt.show()
