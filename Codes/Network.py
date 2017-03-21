# -*- coding: utf-8 -*-

import numpy as np
import Layer as ll

# Fonction Sigmoïde
def sigmoid(x):
    return np.tanh(x)

# Dérivée de la fonction Sigmoïde
def dsigmoid(y):
    return 1.0 - y**2

# Fonction ReLu
def relu(x):
    return np.maximum(x, 0)

class Network:
    """
    Cette classe permet de contruire et utiliser un réseau de neurones
    """
    
    def __init__(self, *layers):
        """
        Construit le réseau de neurones (couches et poids entre couche)
        
        Prend en argument un dictionnaire contenant des couches de neurones
        """
        
        # Récupération du nombre de couches et construction des couches
        self.nbLayers = len(layers)
        self.activations = [layer["activation"] for layer in layers]
        self.layers = [np.ones(layer["dimension"]) for layer in layers]
        
        # Attribution des poids entre les neurones de chaque couche
        self.weights = []
        for i in range(self.nbLayers-1):
            matrix = np.ones([layers[i]["dimension"], layers[i+1]["dimension"]])
            self.weights.append(layers[i]["poids"]*matrix)
        # matrix = np.ones([layers[self.nbLayers-1]["dimension"],1])
        # self.weights.append(layers[self.nbLayers-1]["poids"]*matrix)
        
        self.m = [0 for i in range(len(self.weights))]
    
    def train(self, data, iteration=1000, N=0.5, M=0.1):
        """
        Permet d’entraîner le réseau de neurones, c’est-à-dire à calculer les
        poids entre chaque neurone de chaque couche
        
        Prend en argument les valeurs à attribuer aux neurones de la couche
        d’entrée et celles à attribuer aux neurones de la couche de sortie, le
        nombre de fois où on répète l’entraînement
        """
        for i in range(iteration):
            error = 0.0
            for d in data:
                inputs = d[0]
                targets = d[1]
                self.test(inputs)
                error = error + self.computation(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)
    
    def computation(self, targets, N, M):
        """
        Permet de calculer l’erreur pour une entrée donnée
        
        Prend en argument les valeurs de sortie que l’on veut
        """
        
        # Calcul de l’écart entre la sortie calculée et la sortie théorique
        error = targets - self.layers[-1]
        
        # Initialisation de la liste contenant les termes d’erreurs de chaque couche
        deltas = list()
        
        # Calcul des termes d’erreurs pour la couche de sortie
        de = error * dsigmoid(self.layers[-1])
        deltas.append(de)
        
        # Calcul des termes d’erreurs pour les couches intermédiaires
        for i in range(len(self.layers)-2, 0, -1):
            de = np.dot(deltas[-1], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.append(de)
        
        # Remise dans le bon sens (étant donné qu’on est partie de la sortie)
        deltas.reverse()
        
        # Calcul des nouveaux poids
        for i, j in enumerate(self.weights):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            # Transforme [1, 1] en array([[1, 1]])
             
            dw = np.dot(layer.T, delta)
            self.weights[i] += N*dw + M*self.m[i]
            self.m[i] = dw
        
        # Calcul de l’erreur
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.layers[-1][k])**2
        return error
    
    def test(self, inputs):
        """
        Permet de tester le réseau de neurones, celui-ci ayant été 
        préalablement entraîné (sinon, ça ne sert pas à grand chose :p)
        
        Prend en argument les valeurs à donner aux neurones
        de la couche d’entrée
        """
        
        # Vérification que le nombre d’entrée est compatible avec le réseau
        if len(inputs) != len(self.layers[0]):
            raise ValueError('Le nombre de valeurs d’entrée est incompatible avec le réseau de neurones…')
        
        # Calcul de la valeur de chaque neurone avec les entrées données
        self.layers[0] = inputs
        for i in range(1, self.nbLayers):
            if self.activations[i-1] == "sigmoid":
                self.layers[i] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
            elif self.activations[i-1] == "relu":
                self.layers[i] = relu(np.dot(self.layers[i-1], self.weights[i-1]))
            else:
                raise ValueError('La fonction d’activation n’est pas définie')
        
        # Renvoi de la couche de sortie
        return self.layers[-1]

# Exemple par défaut
L1 = ll.Layer("entree", 2, "sigmoid", 0.3)
L2 = ll.Layer("inter1", 2, "sigmoid", 0.2)
L4 = ll.Layer("sortie", 1, "sigmoid", 0.7)
N = Network(L1.get(), L2.get(), L4.get())
data =[
        [[0, 0], [0]], 
        [[0, 1], [0]],
        [[1, 0], [0]],
        [[1, 1], [1]]
        ]

# Test
for d in data:
    print(N.test(d[0]))

N.train(data)

# Test
for d in data:
    print(N.test(d[0]))