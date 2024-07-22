# lets us first code a neuron
'''this neuron has 3 neurons that it gets inputs from and they 
all have weights and each neuron has a bias. 
Neural network will randomly initialize weights and then back propagation will tweak those weights
'''
inputs = [1, 2, 3, 2.5] # made up values

# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
#             inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
#             inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
# print(output)

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

'''we don't have to have weights and biases necessarily. We can have simple networks with just either of them but they certainly do help'''

layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

'''homologous array - at each dimension it needs to have the same size
l = [[1,3,5],
     [2,3,4]]     shape: (2,4)'''
     # cant have a shape if it is not homologous cause then its either (2,4) or (2,3) - can't say

# a tensor is an object that can be represented as an array 

# learning rate determines how much of the previous knowledge do we wanna keep vs the new knowledge

# we initialize weights in the range (-1, 1) and the tighter the range is, the better. If it is bigger than 1, then as it passes through the neural network, the data becomes  bigger and bigger and kinda explodes. 

