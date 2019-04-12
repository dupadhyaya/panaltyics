# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:01:16 2017 by Dhiraj Upadhyaya
plot3 
http://pythonplot.com/

"""

(mpg['manufacturer']
 .value_counts(sort=False)
 .plot.barh()
 .set_title('Number of Cars by Make')
)



# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

y=[1,2,3]
y
# Prepare the data
x = np.linspace(0, 10, 100)
x
# Plot the data
plt.plot(x, x, label='linear')

# Add a legend
plt.legend()

# Show the plot
plt.show()


import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
plt.xlim(0.5, 4.5)
plt.show()
