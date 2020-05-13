import math


class Neuron:
    def __init__(self, weights, lamb, alpha):
        self.weights = weights
        self.bias = 1
        self.lamb = lamb
        self.alpha = alpha

        self.inputs = []
        self.output = 0

    def __activate(self, value):
        return 1 / (1 + math.exp(-self.lamb * value))

    def run(self, inputs):

        self.inputs = inputs

        if len(self.inputs) != len(self.weights) - 1:
            raise Exception('Wrong number of inputs')

        run_output = 0
        for pos, _input in enumerate(self.inputs):
            run_output = run_output + _input * self.weights[pos]
        run_output = run_output + self.bias * self.weights[-1]

        self.output = self.__activate(run_output)

        return self.output

    def update(self, error):
        theta = 0.5 * self.lamb * (1 - (self.output * self.output))

        delta = error * theta

        sigma = 2 * self.alpha * delta

        previous_errors = [sigma * weight for weight in self.weights]

        for pos, _input in enumerate(self.inputs):
            self.weights[pos] = self.weights[pos] + sigma * _input
        self.weights[-1] = self.weights[-1] + sigma * self.bias

        return previous_errors
