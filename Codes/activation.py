# -*- coding: utf-8 -*-

import numpy as np

# Fonction sigmoïde
def sigmoid(x):
#    return 1.0 / (1.0 + np.exp(-x))
    return np.tanh(x)

# Dérivée de la fonction sigmoïde
def dsigmoid(y):
#    return y * (1.0 - y)
    return 1.0 - y**2

# Fonction ReLU
def relu(x):
    return x * (x > 0)
    
# Dérivée de la fonction ReLU
def drelu(y):
    return 1.0 * (y > 0)
