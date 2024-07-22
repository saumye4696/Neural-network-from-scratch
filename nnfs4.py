import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
import numpy as np

nnfs.init()

# X, y = vertical_data(samples=100, classes=3)
# # print(y)
# print(X)

def f(x):
    return 2*x**2

p2_delta = 0.001
x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)
approximate_derivative = (y2 - y1) / (x2 - x1)
b = y2 - approximate_derivative * x2

def tangent_line(x):
    return approximate_derivative * x + b

# x = np.array(range(5))
# x = np.arange(0, 5, 0.001)
# y = f(x)
# print(y)
x = np.arange(0, 5, 0.001)
y = f(x)
plt.plot(x,y)

to_plot = [x1 - 0.9 , x1, x1 + 0.9 ]

plt.plot(to_plot, [tangent_line(i) for i in to_plot])

print('Approximate derivative for f(x)' ,f'where x = {x1} is {approximate_derivative} ' )

plt.show()