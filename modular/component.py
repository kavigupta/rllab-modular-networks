
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import mul

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
class ConvLayer(Component):
    def __init__(self, name, in_width, in_height, filter_size, in_channels, out_channels, stride):
        super().__init__()
        self.name = name
        self.stride = stride
        with tf.variable_scope(self.name) as scope:
            self.filter = tf.Variable(np.random.rand(filter_size, filter_size, in_channels, out_channels))
            self.bias = tf.Variable(np.random.rand(in_width, in_height, 1))
            self.parameters = [Var(self.filter), Var(self.bias)]
        self.scope = scope
    def run(self, arg):
        return tf.nn.relu(self.bias + tf.nn.conv2d(arg, self.filter, [1, self.stride, self.stride, 1], padding="SAME"), name="output")

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

class ConvNetwork(LayeredNetwork):
    def __init__(self, name, in_width, in_height, filter_size, channel_sizes, output_channels, stride):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            for i, (prev, out) in enumerate(zip([3] + channel_sizes, channel_sizes + [output_channels])):
                self.parameters.append(ConvLayer(name=str(i),
                                                 in_width=in_width, in_height=in_height,
                                                 filter_size=filter_size,
                                                 in_channels=prev, out_channels=out,
                                                 stride=stride))
        self.scope = scope

def flatten(arg):
    first, *rest = (x.value for x in arg.shape)
    return tf.reshape(arg, [first, reduce(mul, rest)])

class ImageNetwork(LayeredNetwork):
    def __init__(self, name, in_width, in_height, filter_size, channel_sizes, output_channels, stride, hidden_layers, hidden_size):
        super().__init__()
        self.name = name
        with tf.name_scope(self.name) as scope:
            conv = ConvNetwork("conv",
                                 filter_size=filter_size,
                                 in_width=in_width, in_height=in_height,
                                 channel_sizes=channel_sizes, output_channels=output_channels, stride=stride)
            fc = FCNetwork("fc", in_width * in_height * output_channels, hidden_layers, hidden_size)
            self.parameters = [conv, flatten, fc]
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
