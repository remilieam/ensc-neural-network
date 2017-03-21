# -*- coding: utf-8 -*-

class Layer:
    """
    Cette classe permet de créer des couches de neurones
    """
    
    def __init__(self, nom, dimension, activation, poids):
        """
        Constructeur
        """
        self._layer = {"nom": nom, "dimension": dimension,
                "activation": activation, "poids": poids}

    def get(self):
        """
        Permet de récupérer le dictionnaire
        """
        return self._layer