# -*- coding: utf-8 -*-

import numpy as np

# Fonction sigmoïde
def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

# Dérivée de la fonction sigmoïde
def derivee_sigmoide(y):
    return y * (1.0 - y)

# Fonction ReLU
def relu(x):
    return x * (x > 0)
    
# Dérivée de la fonction ReLU
def derivee_relu(y):
    return 1.0 * (y > 0)

# Fonction tangente hyperbolique
def tanh(x):
    return np.tanh(x)

# Dérivée de la fonction tangente hyperbolique
def derivee_tanh(y):
    return 1.0 - y**2