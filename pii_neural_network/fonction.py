#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fonctions d’activation et leurs dérivées pour les 
couches de neurones
"""

import numpy as np
__all__ = ['sigmoide', 'relu', 'tanh']

# Fonction sigmoïde
def sigmoide(x):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la fonction sigmoïde
    
    Argument :
    ----------
    
    x : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 5 par la fonction sigmoïde :
    
    >>> sigmoïde(5)
    0.99330714907571527
    
    Pour connaître l’image de [1, 4] par la fonction sigmoïde :
        
    >>> sigmoïde(np.array([1, 4]))
    array([ 0.73105858,  0.98201379])
    """
    return 1.0 / (1.0 + np.exp(-x))

# Dérivée de la fonction sigmoïde
def derivee_sigmoide(y):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la dérivée de la fonction sigmoïde
    
    Argument :
    ----------
    
    y : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 0.5 par la dérivée de la fonction sigmoïde :
    
    >>> derivee_sigmoide(0.5)
    0.25
    
    Pour connaître l’image de [0.1, 0.9] par dérivée de la fonction sigmoïde :
        
    >>> derivee_sigmoide(np.array([0.1, 0.9]))
    array([ 0.09,  0.09])
    """
    return y * (1.0 - y)

# Fonction ReLU
def relu(x):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la fonction ReLU
    
    Argument :
    ----------
    
    x : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 5 par la fonction ReLU :
    
    >>> relu(5)
    5
    
    Pour connaître l’image de [1, 4] par la fonction ReLU :
        
    >>> relu(np.array([-2, 8]))
    array([0, 8])
    """
    return x * (x > 0)
    
# Dérivée de la fonction ReLU
def derivee_relu(y):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la dérivée de la fonction ReLU
    
    Argument :
    ----------
    
    y : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 0.5 par la dérivée de la fonction ReLU :
    
    >>> derivee_relu(0.5)
    1.0
    
    Pour connaître l’image de [0.1, 0.9] par dérivée de la fonction ReLU :
        
    >>> derivee_relu(np.array([-0.1, 0.9]))
    array([ 0.,  1.])
    """
    return 1.0 * (y > 0)

# Fonction tangente hyperbolique
def tanh(x):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la fonction tangente hyperbolique
    
    Argument :
    ----------
    
    x : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 5 par la fonction tangente hyperbolique :
    
    >>> tanh(5)
    0.99990920426259511
    
    Pour connaître l’image de [1, 4] par la fonction tangente hyperbolique :
        
    >>> tanh(np.array([1, 4]))
    array([ 0.76159416,  0.9993293 ])
    """
    return np.tanh(x)

# Dérivée de la fonction tangente hyperbolique
def derivee_tanh(y):
    """
    Calcule l’image d’un nombre ou d’un tableau à n dimensions 
    par la dérivée de la fonction tangente hyperbolique
    
    Argument :
    ----------
    
    y : nombre (entier ou flottant) ou tableau (numpy.ndarray)
    
    Exemples :
    ----------
    
    Pour connaître l’image de 0.5 par la dérivée de la fonction tangente hyperbolique :
    
    >>> derivee_tanh(0.5)
    0.75
    
    Pour connaître l’image de [0.1, 0.9] par dérivée de la fonction tangente hyperbolique :
        
    >>> derivee_tanh(np.array([0.1, 0.9]))
    array([ 0.99,  0.19])
    """
    return 1.0 - y**2
