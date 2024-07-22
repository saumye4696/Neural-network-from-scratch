import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
import numpy as np

nnfs.init()

def f(x):
    return 2*x**2
    
x = np.arange(0, 5, 0.001)
y = f(x)
plt.plot(x,y)

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta
y1 = f(x1)
y2 = f(x2)
print ((x1, y1), (x2, y2))

approximate_derivative = (y2 - y1) / (x2 - x1)
b = y2 - approximate_derivative * x2

def tangent_line ( x ):
    return approximate_derivative * x + b

colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative * x) + b

    

plt.show()