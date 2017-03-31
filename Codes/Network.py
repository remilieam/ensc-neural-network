# -*- coding: utf-8 -*-

import numpy as np
import layer as ll

###############################################################################

# Fonction Sigmoïde
def sigmoid(x):
#    return 1.0 / (1.0 + np.exp(-x))
    return np.tanh(x)

# Dérivée de la fonction Sigmoïde
def dsigmoid(y):
#    return y * (1.0 - y)
    return 1.0 - y**2

# Fonction ReLu
def relu(x):
    return np.maximum(x, 0)
    
# Dérivée de la fonction ReLu
def drelu(y):
    return 1.0 * (y > 0)

###############################################################################
###############################################################################

class Network:
    """
    Cette classe permet de contruire et utiliser un réseau de neurones
    """

###############################################################################

    def __init__(self, *layers):
        """
        Construit le réseau de neurones (couches et poids entre couche)
        
        Prend en argument un dictionnaire contenant des couches de neurones
        """
        # Récupération du nombre de couches et construction des couches
        self.nbLayers = len(layers)
        self.activations = [layer["activation"] for layer in layers]
        self.layers = [np.ones(layers[i]["dimension"] + (i == 0)) for i in range(0, self.nbLayers)]
        
        # Attribution des poids entre les neurones de chaque couche
        self.weights = []
        for i in range(self.nbLayers-1):
            matrix = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights.append(2 * layers[i]["poids"] * matrix - layers[i]["poids"])

###############################################################################

    def train(self, data, iteration = 1000, N = 0.5):
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
                error = error + self.computation(targets, N)
            if ((i+1) % 100) == 0 :
                print('À l’itération', (i+1), 'l’erreur est de : %-.5f' % error)

###############################################################################

    def computation(self, targets, N):
        """
        Permet de calculer l’erreur pour une entrée donnée
        
        Prend en argument les valeurs de sortie que l’on veut et
        le coefficient d’apprentissage N
        """
        # Vérification de la compatibilité
        if len(targets) != len(self.layers[-1]):
            raise ValueError('Le nombre de valeurs de sortie est incompatible avec le réseau de neurones…')
        
        # Calcul de l’écart entre la sortie calculée et la sortie théorique
        error = targets - self.layers[-1]
        
        # Initialisation de la liste contenant les termes d’erreurs de chaque couche
        deltas = list()
        
        # Calcul des termes d’erreurs pour la couche de sortie
        if self.activations[-2] == "sigmoid":
            de = error * dsigmoid(self.layers[-1])
        elif self.activations[-2] == "relu":
            de = error * drelu(self.layers[-1])
        deltas.append(de)
        
        # Calcul des termes d’erreurs pour les couches intermédiaires
        for i in range(self.nbLayers-2, 0, -1):
            if self.activations[i-1] == "sigmoid":
                de = np.dot(deltas[-1], self.weights[i].T) * dsigmoid(self.layers[i])
            elif self.activations[i-1] == "relu":
                de = np.dot(deltas[-1], self.weights[i].T) * drelu(self.layers[i])
            deltas.append(de)
        
        # Remise dans le bon sens (étant donné qu’on est partie de la sortie)
        deltas.reverse()
        
        # Correction des poids
        for i in range(len(self.weights)):
            change = np.dot(np.array([self.layers[i]]).T, np.array([deltas[i]]))
            self.weights[i] += N * change
        
        # Calcul de l’erreur quadratique
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.layers[-1][i])**2
        return error

###############################################################################

    def test(self, inputs):
        """
        Permet de tester le réseau de neurones, celui-ci ayant été 
        préalablement entraîné (sinon, ça ne sert pas à grand chose :p)
        
        Prend en argument les valeurs à donner aux neurones
        de la couche d’entrée
        """
        # Vérification que le nombre d’entrée est compatible avec le réseau
        if len(inputs + [1.0]) != len(self.layers[0]):
            raise ValueError('Le nombre de valeurs d’entrée est incompatible avec le réseau de neurones…')
        
        # Calcul de la valeur de chaque neurone avec les entrées données
        self.layers[0] = np.array(inputs + [1.0])
        for i in range(1, self.nbLayers):
            if self.activations[i-1] == "sigmoid":
                self.layers[i] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
            elif self.activations[i-1] == "relu":
                self.layers[i] = relu(np.dot(self.layers[i-1], self.weights[i-1]))
            else:
                raise ValueError('La fonction d’activation n’est pas définie')
        
        # Renvoi de la couche de sortie
        return self.layers[-1]

###############################################################################
###############################################################################

# Couches
L1 = ll.Layer("entree", 2, "sigmoid", 0.1)
L2 = ll.Layer("inter1", 5, "sigmoid", 0.3)
L3 = ll.Layer("inter2", 3, "sigmoid", 0.2)
L4 = ll.Layer("sortie", 1, "sigmoid", 1)

# Construction du réseau
N = Network(L1.get(), L2.get(), L3.get(), L4.get())

# Données à apprendre
data = [
        [[0, 0], [0]], 
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
        ]

# Test avant l’entraînement
print("Avant :")
for d in data:
    print(d[0], " -> ", N.test(d[0]))

# Entraînement
N.train(data)

# Test après l’entraînement
print("Après :")
for d in data:
    print(d[0], " -> ", N.test(d[0]))