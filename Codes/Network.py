# -*- coding: utf-8 -*-

import numpy as np
import activation as at

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
        self.layers = [np.ones((1, layers[i]["dimension"] + (i == 0))) for i in range(0, self.nbLayers)]
        self.activations = [layer["activation"] for layer in layers]
        
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
                inputs = np.array([d[0]])
                targets = np.array([d[1]])
                self.test(inputs)
                error = error + self.computation(targets, N)
            if ((i+1) % 100) == 0 :
                print("À l’itération", (i+1), "l’erreur est de : %-.5f" %error)

###############################################################################

    def computation(self, targets, N):
        """
        Permet de calculer l’erreur pour une entrée donnée
        
        Prend en argument les valeurs de sortie que l’on veut et
        le coefficient d’apprentissage N
        """
        # Vérification de la compatibilité
        if targets.size != self.layers[-1].size:
            raise ValueError('Le nombre de valeurs de sortie est incompatible avec le réseau de neurones…')
        
        # Initialisation de la liste contenant les termes d’erreurs de chaque couche
        deltas = list()
        
        # Calcul des termes d’erreurs pour la couche de sortie
        if self.activations[-2] == "sigmoid":
            de = (targets - self.layers[-1]) * at.dsigmoid(self.layers[-1])
        elif self.activations[-2] == "relu":
            de = (targets - self.layers[-1]) * at.drelu(self.layers[-1])
        deltas.append(de)
        
        # Calcul des termes d’erreurs pour les couches intermédiaires
        for i in range(self.nbLayers-2, 0, -1):
            if self.activations[i-1] == "sigmoid":
                de = np.dot(deltas[-1], self.weights[i].T) * at.dsigmoid(self.layers[i])
            elif self.activations[i-1] == "relu":
                de = np.dot(deltas[-1], self.weights[i].T) * at.drelu(self.layers[i])
            deltas.append(de)
        
        # Remise dans le bon sens (étant donné qu’on est partie de la sortie)
        deltas.reverse()
        
        # Correction des poids
        for i in range(len(self.weights)):
            change = np.dot(self.layers[i].T, deltas[i])
            self.weights[i] += N * change
        
        # Calcul de l’erreur quadratique
        error = 0.0
        for i in range(targets.size):
            error = error + 0.5 * (targets[0][i] - self.layers[-1][0][i])**2
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
        if (inputs.size + 1) != self.layers[0].size:
            raise ValueError("Le nombre de valeurs d’entrée est incompatible avec le réseau de neurones…")
        
        # Calcul de la valeur de chaque neurone avec les entrées données
        self.layers[0] = np.array([np.append(inputs, [1.])])
        for i in range(1, self.nbLayers):
            if self.activations[i-1] == "sigmoid":
                self.layers[i] = at.sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
            elif self.activations[i-1] == "relu":
                self.layers[i] = at.relu(np.dot(self.layers[i-1], self.weights[i-1]))
            else:
                raise ValueError("La fonction d’activation n’est pas définie")
        
        # Renvoi de la couche de sortie
        return self.layers[-1]
