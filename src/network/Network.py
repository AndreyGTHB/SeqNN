import numpy as np


class Neuron:
    def __init__(self, type, activation_function):
        self.type = type
        self.income = 0
        self.activation = 0
        self.activation_function = activation_function
        self.error = 0
        self.transitions = []

    def __str__(self):
        return 'Neuron {' + f'type: {self.type}, income: {self.income}, activation: {self.activation},' \
                            f' transitions: {self.transitions}'

    def activate(self):
        self.activation = self.activation_function(self.income)

    def join(self, target, weight=round(np.random.uniform(-0.5, 0.5), 2)):
        if self.type == 'output':
            raise TypeError('Output neurons can not join another neurons.')
        self.transitions.append(Transition(self, target, weight))

    def process(self):
        if self.type == 'output':
            return self.activation
        for t in self.transitions:
            t.sendImpulseToTarget(self.activation)
        self.activation = 0

    def back_process(self):
        if self.type == 'input':
            return self.error
        for t in self.transitions:
            t.sendErrorToSource(self.error)


class Transition:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight

    def sendImpulseToTarget(self, impulse):
        self.target.income += impulse * self.weight

    def sendErrorToSource(self, impulse):
        self.source.Error = impulse * self.weight


class Network:
    def __init__(self, input_neurons, activation_function):
        self.layers = [[]]
        self.activation_function = activation_function
        for i in range(input_neurons):
            self.layers[-1].append(Neuron('input', activation_function))

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
        output_data = []
        if self.layers[-1][0].type != 'output':
            raise RuntimeError('Network have not output layer')
        for i in range(len(self.layers[0])):
            neuron = self.layers[0][i]
            neuron.income = input_data[i]
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
            for index_of_training_data in range(len(training_input)):
                input_data = training_input[index_of_training_data]
                output_data = training_output[index_of_training_data]
                prediction = self.predict(input_data)
                back_layers = self.layers[::-1]
                for i in range(len(back_layers[0])):
                    back_layers[0][i].error = output_data[i] - prediction[i]
                for layer in back_layers:
                    for neuron in layer:
                        pass


