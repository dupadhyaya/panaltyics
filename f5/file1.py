# Load CSV
import numpy
filename = 'pima-indians-diabetes-data.csv'
raw_data = open(filename, 'rb')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)
