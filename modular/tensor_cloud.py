
import tensorflow as tf
import numpy as np

from component import Concatenation

class TensorCloud:
    @staticmethod
    def product(first, second, concat, pred = lambda x, y: True):
        return TensorCloud({
            (k1, k2) : concat(v1, v2)
                for k1, v1 in first
                for k2, v2 in second
                    if pred(k1, k2)
        }, first.components | second.components)
    @staticmethod
    def input(size, name):
        return TensorCloud({() : input_tensor(size, name)})
    def __init__(self, params_to_tensor, components=set()):
        self.params_to_tensor = {(k,) : v for k, v in params_to_tensor.items()}
        self.components = set(components)
    def __or__(self, nets):
        if not isinstance(nets, dict):
            nets = {() : nets}
        values = {}
        all_nets = set()
        for net_spec, net in nets.items():
            for params, tensor in self:
                if isinstance(net, type(lambda: None)):
                    net_to_use = net(params)
                else:
                    net_to_use = net
                all_nets |= {net_to_use}
                values[(params,) + (net_spec,)] = net_to_use(tensor)
        return TensorCloud(values, self.components | all_nets)
    def label(self, name):
        return TensorCloud({k + (name,) : v for k, v in self.params_to_tensor.items()}, self.components)
    def __add__(self, other):
        vals = {}
        vals.update(dict(self))
        vals.update(dict(other))
        return TensorCloud(vals, self.components | other.components)
    def __iter__(self):
        return iter((simplify(x), y) for x, y in self.params_to_tensor.items())
    def __repr__(self):
        return f"TensorCloud({self.params_to_tensor}, {self.components})"

def input_tensor(size, name):
    if isinstance(size, int):
        size = [size]
    return tf.placeholder(np.float32, [None] + size, name=name)

def flatten(tup):
    result = ()
    for x in tup:
        if isinstance(x, tuple):
            result += flatten(x)
        else:
            result = result + (x,)
    return result

def simplify(x):
    x = flatten(x)
    if len(x) == 1:
        return x[0]
    return x
