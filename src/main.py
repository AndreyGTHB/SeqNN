from network.Network import *


def leaky_reLU(x):
    return max(x / 10, x)


def derivative_of_leaky_reLU(x):
    if x > 0:
        return 1
    return 0.1


network = Network(3, leaky_reLU, derivative_of_leaky_reLU)
network.add_layer(2, 'output')

inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
outputs = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1]]
network.train(inputs, outputs, 80, 0.02)

print(network.predict([0, 0, 0]))
