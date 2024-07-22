import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

# np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]
X, y = spiral_data(100, 3)
plt.scatter(X[:,0], X[:,1], c=y, s=40,cmap=plt.cm.Spectral)
plt.show()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # randn takes the shape that we wanna pass - size of the input coming in and how many neurons we wanna have. inputs will be coming in a batch but here single sample is 4 so we say n_inputs.
        # randn is a gaussian distribution bounded around 0
        self.biases = np.zeros((1, n_neurons))

    # The input to np.zeros - the first parameter is actually the shape that's why we made it one parameter vs randn where the parameters are the shape

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
# layer1 = Layer_Dense(4, 5)
layer1 = Layer_Dense(2,5)
layer2 = Layer_Dense(5, 2)

activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

# diff between step and sigmoid function - since sigmoid is granular, we can find how close were we to output a zero in case it outputs a 1 and so we cant know the loss. Same things happen when we take the sigmoid and Relu function - sigmoid has the issue called as the vanishing gradient problem
# Relu - fast (sigmoid is fast but not as fast as Relu), simple and just works. - most popular for hidden layers.
# why use AF at all - 

# nnfs.init() will do the np.random.seed thing and all set the datatype for numpy dot product otherwise it will take some different values and then we get slightly different values. 

# Input (which is the output layer of data which is being input into the activation function ) -> exponentiate -> Normalize -> output
# exponentiate + normaliza = softmax