
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
    def __repr__(self):
        return "<Component {type(self).__name__} {parameters}>".format(parameters=self.parameters)

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
    def __repr__(self):
        return "<Var {variable}>".format(variable=self.variable)

def gen_variable(*dims):
    return tf.random_normal(dims, stddev=0.01)

def gen_variable_filter(*filter_shape):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    magnitude = 4*np.sqrt(6.0/(fan_in + fan_out))
    low = -magnitude
    high = magnitude
    return np.array(np.random.uniform(low, high, size=filter_shape), dtype=np.float32)

class Layer(Component):
    def __init__(self, name, start_size, output_size):
        super().__init__()
        self.name = name
        with tf.variable_scope(self.name) as scope:
            self.layer = tf.Variable(gen_variable(start_size, output_size) - 0.5, name="layer", trainable=True)
            self.bias = tf.Variable(-gen_variable(1, output_size), trainable=True)
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
            self.filter = tf.Variable(gen_variable_filter(filter_size, filter_size, in_channels, out_channels), trainable=True)
            self.bias = tf.Variable(gen_variable(in_width, in_height, 1), trainable=True)
            self.parameters = [Var(self.filter), Var(self.bias)]
        self.scope = scope
    def run(self, arg):
        return tf.nn.relu(self.bias + tf.nn.conv2d(arg, self.filter, [1, self.stride, self.stride, 1], padding="SAME"), name="output")

class LayeredNetwork(Component):
    def __init__(self, name, *layers):
        super().__init__()
        self.name = name
        with tf.name_scope(self.name) as scope:
            self.parameters = [layer() for layer in layers]
        self.scope = scope
    def run(self, arg):
        val = arg
        for param in self.parameters:
            val = param(val)
        return val

class FCNetwork(LayeredNetwork):
    def __init__(self, name, start_dim, inter_sizes, output_size):
        parameters = []
        for i, (prev, out) in enumerate(zip([start_dim] + inter_sizes, inter_sizes + [output_size])):
            parameters.append(lambda i=i, prev=prev, out=out: Layer(str(i), prev, out))
        super().__init__(name, *parameters)

class ConvNetwork(LayeredNetwork):
    def __init__(self, name, in_width, in_height, filter_size, channel_sizes, output_channels, stride):
        parameters = []
        for i, (prev, out) in enumerate(zip([3] + channel_sizes, channel_sizes + [output_channels])):
            parameters.append(lambda i=i, prev=prev, out=out: ConvLayer(name=str(i),
                                             in_width=in_width, in_height=in_height,
                                             filter_size=filter_size,
                                             in_channels=prev, out_channels=out,
                                             stride=stride))
        super().__init__(name, *parameters)

class EmptyComponent(Component):
    def __init__(self):
        super().__init__()
    def initialize(self, sess):
        pass
    def read(self, sess, values):
        pass
    def write(self, sess):
        pass
    def __repr__(self):
        return str(type(self)) + "()"

class Flatten(EmptyComponent):
    def __call__(self, arg):
        first, *rest = (x.value for x in arg.shape)
        return tf.reshape(arg, [first if first is not None else -1, reduce(mul, rest)])

class ImageNetwork(LayeredNetwork):
    def __init__(self, name, in_width, in_height, filter_size, channel_sizes, output_channels, stride, hidden_layers, hidden_size):
        conv = lambda: ConvNetwork("conv",
                             filter_size=filter_size,
                             in_width=in_width, in_height=in_height,
                             channel_sizes=channel_sizes, output_channels=output_channels, stride=stride)
        fc = lambda: FCNetwork("fc", in_width * in_height * output_channels, hidden_layers, hidden_size)
        super().__init__(name, conv, Flatten, fc)

class Concatenation(Component):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            self.scope = scope
    def run(self, first, second):
        if second.shape[0].value is not None:
            second = tf.tile(second, [tf.shape(first)[0], 1])
        return tf.concat([first, second], axis=1)

class Addition(Component):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            self.scope = scope
    def run(self, first, second):
        return tf.add(first, second)

class Loss(Component):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.parameters = []
        with tf.name_scope(self.name) as scope:
            self.scope = scope
    def run(self, first, second):
        return tf.nn.l2_loss(first - second)
