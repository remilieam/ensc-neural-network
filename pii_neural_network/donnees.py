#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fonction pour la récupération de données d’apprentissage pour un réseau
de neurones
"""

import os
import struct
import numpy as np
__all__ = ['recuperation']

def recuperation(chemin_donnees):
    """
    Permet de récupérer la base de données MNIST composées de 4 fichiers
    
    Les 4 fichiers doivent avoir les extensions .idx3-ubyte
    et .idx1-ubyte
    
    Argument :
    ----------
    
    chemin_donnees : chaîne de caractères donnant le chemin relatif du dossier
                     contenant les 4 fichiers de la base de données MNIST
    
    Exemple :
    ---------
    
    Si les 4 fichiers sont rangés dans le dossier 'MNIST_data',
    il faut appeler la fonction de la sorte :
    
    >>> recuperation('MNIST_data')
    """
    # Récupération des jeux de données pour l’entraînement (images et labels)
    jeu_entrainement_images = os.path.join(chemin_donnees, "train-images.idx3-ubyte")
    jeu_entrainement_labels = os.path.join(chemin_donnees, "train-labels.idx1-ubyte")

    # Construction des tableaux contenant les jeux de données pour l’entraînement
    with open(jeu_entrainement_images, "rb") as fichier:
        nb_maqique, nb_images, nb_lignes, nb_colonnes = struct.unpack(">IIII", fichier.read(16))
        entrainement_images = np.fromfile(fichier, dtype = np.uint8).reshape(60000, 1, 28, 28)
    with open(jeu_entrainement_labels, "rb") as fichier:
        nb_magique, nb_labels = struct.unpack(">II", fichier.read(8))
        entrainement_labels = np.fromfile(fichier, dtype = np.uint8)

    # Récupération des jeux de données pour les tests (images et labels)
    jeu_test_images = os.path.join(chemin_donnees, "t10k-images.idx3-ubyte")
    jeu_test_labels = os.path.join(chemin_donnees, "t10k-labels.idx1-ubyte")

    # Construction des tableaux contenant les jeux de données pour les tests
    with open(jeu_test_images, "rb") as fichier:
        nb_maqique, nb_images, nb_lignes, nb_colonnes = struct.unpack(">IIII", fichier.read(16))
        test_images = np.fromfile(fichier, dtype = np.uint8).reshape(10000, 1, 28, 28)
    with open(jeu_test_labels, "rb") as fichier:
        nb_magique, nb_labels = struct.unpack(">II", fichier.read(8))
        test_labels = np.fromfile(fichier, dtype = np.uint8)

    # Création des matrices colonnes solutions (tous les coefficients sont
    # à 0 sauf le coefficient correspondant à la solution)
    # Par exemple pour la solution 5 : [0 0 0 0 0 1 0 0 0 0]
    labels = dict()
    for i in range(10):
        label = np.zeros((10, 1), dtype = np.uint8)
        label[np.uint8(i)] = 1
        labels[np.uint8(i)] = label

    # Mise en forme des jeux de données pour l’entraînement
    # (normalisation des valeurs des pixels et construction d’une matrice
    # colonne pour les labels)
    e_images = np.array([image/255.0 for image in entrainement_images])
    e_labels = np.array([labels[label] for label in entrainement_labels]).astype(np.uint8)

    # Mise en forme des jeux de données pour les tests
    # (normalisation des valeurs des pixels et construction d’une matrice
    # colonne pour les labels)
    t_images = np.array([image/255.0 for image in test_images]).astype(np.float32)
    t_labels = np.array([labels[label] for label in test_labels]).astype(np.uint8)

    # Renvoi des jeux de donnees
    return (e_images, e_labels), (t_images, t_labels)
