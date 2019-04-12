# WORKING
import numpy as np
import matplotlib.pyplot as plt

points = np.arange(-5, 5, 0.01)
dx, dy = np.meshgrid(points, points)
z = (np.sin(dx)+np.sin(dy))
plt.imshow(z)
plt.colorbar()
plt.title('plot for sin(x)+sin(y)')
plt.show()