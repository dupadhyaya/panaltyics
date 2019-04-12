# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:07:32 2017 by Dhiraj Upadhyaya
plot and pandas

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(pd.__version__)
#http://pbpython.com/extras/sample-salesv2.csv - csv file
#%matplotlib inline
sales=pd.read_csv("./data/sample-salesv2.csv",parse_dates=['date'])
sales.head()
sales.describe()
"""
We can tell that customers on average purchases 10.3 items per transaction
The average cost of the transaction was $579.84
It is also easy to see the min and max so you understand the range of the data
"""
sales['unit price'].describe()
#average price is $56.18 but it ranges from $10.06 to $99.97.
#showing the output of dtypes :  date column is a datetime field. I also scan this to make sure that any columns that have numbers are floats or ints so that I can do additional analysis in the future.
sales.dtypes

# Plotting some data
customers = sales[['name','ext price','date']]
customers.head()

#In order to understand purchasing patterns, let’s group all the customers by name. We can also look at the number of entries per customer to get an idea for the distribution.

customer_group = customers.groupby('name')
print(customer_group)
customer_group.size()
#customers
#data is in a simple format to manipulate, let’s determine how much #each customer purchased during our time frame.

#sum function allows us to quickly sum up all the values by customer. #We can also sort the data using the sort command.

sales_totals = customer_group.sum()
sales_totals
sales_totals.sort_values(['ext price'], ascending=True).head()

my_plot = sales_totals.plot(kind='bar')

"""
sorting the data in descending order
removing the legend
adding a title
labeling the axes
"""
my_plot = sales_totals.sort_values(['ext price'], ascending=False).plot( kind='bar', legend=None, title= "Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales ($)")


customers = sales[['name','category','ext price','date']]
customers.head()
category_group=customers.groupby(['name','category']).sum()
category_group.head()
#he category representation looks good but we need to break it apart to graph it as a stacked bar graph. unstack can do this for us.

category_group.unstack().head()

my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")

#clean this up a little bit, we can specify the figure size and customize the legend.
my_plot = category_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Customer",figsize=(9, 7))
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales")
my_plot.legend(["Total","Belts","Shirts","Shoes"], loc=9,ncol=4)

"""
Now that we know who the biggest customers are and how they purchase products, we might want to look at purchase patterns in more detail.

Let’s take another look at the data and try to see how large the individual purchases are. A histogram allows us to group purchases together so we can see how big the customer transactions are.
"""
purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()

#create a histogram with 20 bins to show the distribution of purchasing patterns.

purchase_plot = purchase_patterns['ext price'].hist(bins=20)
purchase_plot.set_title("Purchase Patterns")
purchase_plot.set_xlabel("Order Amount($)")
purchase_plot.set_ylabel("Number of orders")
plt.show()

#-----
purchase_patterns = sales[['ext price','date']]
purchase_patterns.head()
purchase_patterns = purchase_patterns.set_index('date')
purchase_patterns.head()

purchase_patterns.resample('M',how=sum)
#M’ as the period for resampling which means the data should be resampled on a month boundary.
purchase_plot = purchase_patterns.resample('M',how=sum).plot(title="Total Sales by Month",legend=None)

fig = purchase_plot.get_figure()
fig.savefig("total-sales.png")
