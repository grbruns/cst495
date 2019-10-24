# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:59:59 2019

@author: Glenn
"""

import math
import random

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v,w))

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def forward(layer, inputs):
    inputs_with_bias = inputs + [1]
    outputs = [sigmoid(dot(neuron, inputs_with_bias)) for neuron in layer]
    return outputs

def backprop(layer, inputs, targets, learning_rate=0.5):
    outputs = forward(layer, inputs)
    output_deltas = [ (output - target) * output * (1 - output) 
                for output, target in zip(outputs, targets) ]
    
    for i, neuron in enumerate(layer):
        deltaz_w = [ output_deltas[i] * x for x in inputs + [1] ]
        for j in range(len(neuron)):
            neuron[j] -= learning_rate*deltaz_w[j]

random.seed(0)
inputs = [1, 3]
num_neurons = 2
layer = [[random.random() for __ in range(len(inputs)+1)]
        for __ in range(num_neurons) ]
targets = [0.7, 0.3]
for i in range(100):
    backprop(layer, inputs, targets)
    
print('trained output: {:.5f}, {:.5f}'.format(*forward(layer, inputs+[1])))