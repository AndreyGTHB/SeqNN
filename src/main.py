from network.Network import *


network = Network(3, lambda x: 1 / (1 + 2.718 ** -x))
network.add_layer(1, 'output')

print(network.predict([1, 0, 1]))
