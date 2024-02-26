import numpy as np



class Neuron:
    def __init__(self, input_size):
        # Initialize weights randomly and bias as 0
        self.weights = np.random.randn(input_size)
        self.bias = 0
        
    def forward(self, inputs):
        # Perform weighted sum of inputs and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function (here using sigmoid)
        output = self.sigmoid(weighted_sum)
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights randomly and bias as 0 for each neuron in the layer
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)
        
    def forward(self, inputs):
        # Perform matrix multiplication of inputs and weights, then add biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)


# Number of data points
num_points = 100

# Number of features (input dimensions)
num_features = 2

# Generate random input data
X = np.random.rand(num_points, num_features)

# Generate random output labels
y = np.random.randint(0, 2, size=num_points)  # Assuming binary classification (0 or 1)

print("Random input data (X):\n", X)
print("\nRandom output labels (y):\n", y)

layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)

print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)