
from abc import ABCMeta, abstractmethod
from functools import reduce

import tensorflow as tf
import numpy as np

class Component:
    def __init__(self):
        self.inputs_outputs = {}
    def initialize(self, sess):
        for param in self.parameters:
            param.initialize(sess)
    def read(self, sess, values):
        for param in self.parameters:
            param.read(values)
    def write(self, sess):
        for param in self.parameters:
            yield from param.write(sess)
    def __call__(self, *args):
        with tf.variable_scope(self.scope):
            if args not in self.inputs_outputs:
                self.inputs_outputs[args] = self.run(*args)
            return self.inputs_outputs[args]

class Var(Component):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable
    def initialize(self, sess):
        sess.run(tf.variables_initializer([self.variable]))
    def read(self, sess, values):
        sess.run(tf.assign(self.variable, values[self.variable.name]))
    def write(self, sess):
        yield (self.variable.name, sess.run(self.variable))

class Layer(Component):
    def __init__(self, name, start_size, output_size):
        super().__init__()
        self.name = name
        with tf.variable_scope(self.name) as scope:
            self.layer = tf.Variable(np.random.rand(start_size, output_size) - 0.5, name="layer")
            self.bias = tf.Variable(-np.random.rand(1, output_size))
            self.parameters = [Var(self.layer), Var(self.bias)]
        self.scope = scope
    def run(self, arg):
        return tf.nn.relu(self.bias + tf.matmul(arg, self.layer, name="output"))

class LayeredNetwork(Component):
    def run(self, arg):
        val = arg
        for param in self.parameters:
            val = param(val)
        return val

class FCNetwork(LayeredNetwork):
    def __init__(self, name, start_dim, inter_sizes, output_size):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            for i, (prev, out) in enumerate(zip([start_dim] + inter_sizes, inter_sizes + [output_size])):
                self.parameters.append(Layer(str(i), prev, out))
        self.scope = scope

class Concatenation(Component):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            self.scope = scope
    def run(self, first, second):
        return tf.concat([first, second], axis=1)
