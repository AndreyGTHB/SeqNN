from network.Network import *


def sigmoid(x):
    return 1 / (1 + 2.718 ** -x)


def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


network = Network(3, sigmoid, derivative)
network.add_layer(2, 'output')

inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
outputs = [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]
network.train(inputs, outputs, 1000, 0.03)

print(network.predict([0, 1, 0]))
