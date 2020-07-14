import numpy as np


class Neuron:
    def __init__(self, type, activation_function):
        self.type = type
        self.income = {'impulse': 0, 'error': 0}
        self.activation = 0
        self.activation_function = activation_function
        self.transitions = {'outgoing': [], 'inbox': []}

    def __str__(self):
        return 'Neuron {' + f'type: {self.type}, income: {self.income}, activation: {self.activation},' \
                            f' transitions: {self.transitions}'

    def activate(self):
        self.activation = self.activation_function(self.income['impulse'])

    def join(self, target, weight=round(np.random.uniform(-0.5, 0.5), 2)):
        if self.type == 'output':
            raise TypeError('Output neurons can not join another neurons.')
        t = Transition(self, target, weight)
        self.transitions['outgoing'].append(t)
        target.transitions['inbox'].append(t)

    def process(self):
        if self.type == 'output':
            return self.activation
        for t in self.transitions['outgoing']:
            t.sendImpulseToTarget(self.activation)

    def back_process(self):
        if self.type == 'input':
            return self.income['error']
        for t in self.transitions['inbox']:
            t.sendErrorToSource(self.income['error'])


class Sensor(Neuron):
    def __init__(self):
        super(Sensor, self).__init__('input', lambda x: x)


class Transition:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight

    def sendImpulseToTarget(self, impulse):
        self.target.income['impulse'] += impulse * self.weight

    def sendErrorToSource(self, impulse):
        self.source.income['error'] += impulse * self.weight


class Network:
    def __init__(self, input_neurons, activation_function, derivative_function):
        self.layers = [[]]
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        for i in range(input_neurons):
            self.layers[-1].append(Sensor())

    def add_layer(self, neurons, layer_type='hidden'):
        if neurons <= 0:
            return
        self.layers.append([])
        for i in range(neurons):
            neuron = Neuron(layer_type, self.activation_function)
            self.layers[-1].append(neuron)
            for n in self.layers[-2]:
                n.join(neuron)

    def predict(self, input_data):
        self.clean()
        output_data = []
        if self.layers[-1][0].type != 'output':
            raise RuntimeError('Network have not output layer')
        for i in range(len(self.layers[0])):
            neuron = self.layers[0][i]
            neuron.income['impulse'] = input_data[i]
        for layer in self.layers:
            for neuron in layer:
                neuron.activate()
                neuron.process()
        for neuron in self.layers[-1]:
            activation = neuron.process()
            output_data.append(activation)
        return output_data

    def train(self, training_input, training_output, epochs, learning_rate):
        for epoch in range(epochs):
            for i_of_training_data in range(len(training_input)):
                outputs = training_output[i_of_training_data]
                inputs = training_input[i_of_training_data]
                prediction = self.predict(inputs)
                reversed_network = self.layers[::-1]
                for i in range(len(reversed_network[0])):
                    reversed_network[0][i].income['error'] = outputs[i] - prediction[i]
                for layer in reversed_network[:-1]:
                    for neuron in layer:
                        neuron.back_process()
                for layer in self.layers[1:]:
                    for neuron in layer:
                        for t in neuron.transitions['inbox']:
                            t.weight += neuron.income['error'] * self.derivative_function(neuron.activation) * t.source.activation * learning_rate

    def clean(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.income['impulse'] = 0
                neuron.income['error'] = 0
                neuron.activation = 0

