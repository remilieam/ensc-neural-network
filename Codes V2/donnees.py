# -*- coding: utf-8 -*-

import os
import struct
import numpy as np

class Donnees:
    def __init__(self, chemin_donnees):
        # Récupération des jeux de données pour l’entraînement (images et labels)
        jeu_entrainement_images = os.path.join(chemin_donnees, "train-images.idx3-ubyte")
        jeu_entrainement_labels = os.path.join(chemin_donnees, "train-labels.idx1-ubyte")

        # Construction des tableaux contenant les jeux de données pour l’entraînement
        with open(jeu_entrainement_images, "rb") as fichier:
            nb_maqique, nb_images, nb_lignes, nb_colonnes = struct.unpack(">IIII", fichier.read(16))
            self.entrainement_images = np.fromfile(fichier, dtype = np.uint8).reshape(60000, 1, 28, 28)
        with open(jeu_entrainement_labels, "rb") as fichier:
            nb_magique, nb_labels = struct.unpack(">II", fichier.read(8))
            self.entrainement_labels = np.fromfile(fichier, dtype = np.uint8)

        # Récupération des jeux de données pour les tests (images et labels)
        jeu_test_images = os.path.join(chemin_donnees, "t10k-images.idx3-ubyte")
        jeu_test_labels = os.path.join(chemin_donnees, "t10k-labels.idx1-ubyte")

        # Construction des tableaux contenant les jeux de données pour les tests
        with open(jeu_test_images, "rb") as fichier:
            nb_maqique, nb_images, nb_lignes, nb_colonnes = struct.unpack(">IIII", fichier.read(16))
            self.test_images = np.fromfile(fichier, dtype = np.uint8).reshape(10000, 1, 28, 28)
        with open(jeu_test_labels, "rb") as fichier:
            nb_magique, nb_labels = struct.unpack(">II", fichier.read(8))
            self.test_labels = np.fromfile(fichier, dtype = np.uint8)

        # Création des matrices colonnes solutions (tous les coefficients sont
        # à 0 sauf le coefficient correspondant à la solution)
        # Par exemple pour la solution 5 : [0 0 0 0 0 1 0 0 0 0]
        self.labels = dict()
        for i in range(10):
            label = np.zeros((10, 1), dtype = np.uint8)
            label[np.uint8(i)] = 1
            self.labels[np.uint8(i)] = label

    def recuperation(self):
        # Mise en forme des jeux de données pour l’entraînement
        # (normalisation des valeurs des pixels et construction d’une matrice
        # colonne pour les labels)
        entrainement_images = np.array([image/255.0 for image in self.entrainement_images])
        entrainement_labels = np.array([self.labels[label] for label in self.entrainement_labels]).astype(np.uint8)

        # Mise en forme des jeux de données pour les tests
        # (normalisation des valeurs des pixels et construction d’une matrice
        # colonne pour les labels)
        test_images = np.array([image/255.0 for image in self.test_images]).astype(np.float32)
        test_labels = np.array([self.labels[label] for label in self.test_labels]).astype(np.uint8)

        # Renvoi des jeux de donnees
        return (entrainement_images, entrainement_labels), (test_images, test_labels)
