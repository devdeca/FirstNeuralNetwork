from neuron import Neuron
from utils import absolute
import random


class Network:
    def __init__(self, neurons_quantity, lamb, alpha, threshold, weights):
        self.first_layer = [Neuron(weights[n], lamb, alpha) for n in range(neurons_quantity)]
        self.last_layer = Neuron(weights[-1], lamb, alpha)
        self.threshold = threshold

    def run(self, inputs):
        layer_outputs = []
        for n in self.first_layer:
            layer_outputs.append(n.run(inputs))

        last_output = self.last_layer.run(layer_outputs)

        return last_output

    def update(self, inputs, expected):
        layer_outputs = []
        for n in self.first_layer:
            layer_outputs.append(n.run(inputs))

        last_output = self.last_layer.run(layer_outputs)

        hidden_layer_errors = self.last_layer.update(expected - last_output)

        for pos, n in enumerate(self.first_layer):
            n.update(hidden_layer_errors[pos])
            print(f'Neurônio {pos}: {n.weights}')
        print(f'Neurônio final: {self.last_layer.weights}')

    def train(self, data):

        print('=== Pesos iniciais ===')
        for pos, n in enumerate(self.first_layer):
            print(f'Neurônio {pos}: {n.weights}')
        print(f'Neurônio final: {self.last_layer.weights}')

        error = 1
        it = 0
        while error > self.threshold:
            it = it + 1
            print(f'=== Iteração {it} ===')
            rand = random.randint(0, 3)
            self.update(data[rand][0:2], data[rand][2])
            total = 0
            for d in data:
                total = total + absolute(d[2] - self.run(d[0:2]))
            error = total / len(data)
            print(f'Erro: {error}')
