#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pii_neural_network.couche as c
import numpy as np
__all__ = ['Reseau']

"""
Classes pour créer des réseaux de neurones
(perceptron multi-couches ou réseau neuronal convolutif)
"""

class Reseau:
    """
    Classe pour créer, entraîner et tester un réseau de neurones
    """
    def __init__(self, *couches):
        """
        Constructeur
        
        Argument(s) :
        -------------
        
        *couches : liste d’instances de couches permettant de créer
                   notre réseau, doit commencer obligatoirement par une
                   instance de CoucheEntree et se terminer par une instance
                   de CoucheConnectee
        
        Exemple :
        ---------
        
        Pour créer un réseau ayant 4 couches (couche_1, couche_2, couche_3
        et couche_4) :
        
        >>> reseau = Reseau(couche_1, couche_2, couche_3, couche_4)
        """
        assert len(couches) > 0
        assert isinstance(couches[0], c.CoucheEntree)
        self.couche_entree = couches[0]
        assert isinstance(couches[-1], c.CoucheConnectee)
        self.couche_sortie = couches[-1]
        self.couches = [(couche_precedente, couche) for couche_precedente, couche in zip(couches[:-1], couches[1:])]
        for couche_precedente, couche in self.couches:
            couche.connexion(couche_precedente)

    def prediction(self, entrees):
        """
        Calcule les valeurs de chaque neurone du réseau à partir des valeurs
        à attribuer à la première couche du réseau
        
        Argument :
        ----------
        
        entrees : tableau contenant les valeurs à attribuer à la première
                  couche du réseau
        """
        self.couche_entree.neurones = entrees
        for couche_precedente, couche in self.couches:
            couche.prediction(couche_precedente)

    def propagation(self, donnees, coef_apprentissage):
        """
        Ajuste tous les poids et tous les biais entre chaque couche du réseau
        par rétro-propagation de l’erreur (ou du gradient)
        
        Calcule également l’erreur quadratique pour chaque couple de données
        
        Arguments :
        -----------
        
        donnees            : tableau contenant les données d’apprentissage
                             (valeurs à attribuer aux neurones de la première
                             couches et valeurs désirées des neurones de la
                             dernière couche)
        coef_apprentissage : coefficient d’apprentissage
        
        Retour :
        --------
        
        erreur : somme des erreurs quadratiques pour chaque couple de données
        """
        erreur = 0.0
        dico_nouveaux_poids = {couche: np.zeros_like(couche.poids) for _, couche in self.couches}
        dico_nouveaux_biais = {couche: np.zeros_like(couche.biais) for _, couche in self.couches}
        for entrees, sorties in donnees:
            self.prediction(entrees)
            erreur = erreur + np.sum(0.5*(sorties - self.couche_sortie.neurones)**2)
            delta = (sorties - self.couche_sortie.neurones)*self.couche_sortie.derivee_activation(self.couche_sortie.neurones)
            for couche_precedente, couche in reversed(self.couches):
                nouveaux_poids, nouveaux_biais, delta_precedent = couche.propagation(couche_precedente, delta)
                dico_nouveaux_poids[couche] += nouveaux_poids
                dico_nouveaux_biais[couche] += nouveaux_biais
                delta = delta_precedent
        for _, couche in self.couches:
            couche.poids = couche.poids + coef_apprentissage*dico_nouveaux_poids[couche]/len(donnees)
            couche.biais = couche.biais + coef_apprentissage*dico_nouveaux_biais[couche]/len(donnees)
        return erreur

    def entrainement(self, donnees, nb_entrainements, coef_apprentissage):
        """
        Entraîne le réseau en utilisant la rétro-propagation de l’erreur
        
        Affiche 10 fois (si plus de 10 entraînements) l’erreur quadratique
        afin de voir l’évolution de l’apprentissage
        
        Arguments :
        -----------
        
        donnees            : tableau contenant les données d’apprentissage
                             (valeurs à attribuer aux neurones de la première
                             couches et valeurs désirées des neurones de la
                             dernière couche)
        nb_entrainements   : nombre de fois où le réseau va apprendre les
                             données
        coef_apprentissage : coefficient d’apprentissage
        """
        for i in range(nb_entrainements):
            erreur = self.propagation(donnees, coef_apprentissage)
            if ((i+1) % (nb_entrainements/10)) == 0 :
                    print("À l'itération", (i+1), "l'erreur est de : %-.5f" %erreur)

    def test(self, donnees):
        """
        Test le réseau et affiche pour chaque donnée : les valeurs d’entrées,
        les valeurs de sorties obtenues et les valeurs de sorties souhaitées
        
        Argument :
        ----------
        
        donnees            : tableau contenant les données d’apprentissage
                             (valeurs à attribuer aux neurones de la première
                             couches et valeurs désirées des neurones de la
                             dernière couche)
        
        Retour :
        --------
        
        sorties_predites : tableau contenant les sorties prédites pour chaque 
                           entrée après calcul par le réseau
        """
        sorties_predites = np.zeros(len(donnees))
        donnees_finales = np.copy(donnees)
        for i, (entrees, _) in enumerate(donnees_finales):
            self.prediction(entrees)
            donnees_finales[i][1] = self.couche_sortie.neurones
        return donnees_finales


if __name__ == "__main__":
    import pii_neural_network.fonction as f
    import pii_neural_network.donnees as d
    import matplotlib.pyplot as p

    # Perceptron multi-couches
    xor_couche_entree = c.CoucheEntree(2, 1)
    xor_couche_cachee = c.CoucheConnectee(10, 0.5, f.sigmoide)
    xor_couche_cachee_2 = c.CoucheConnectee(6, 0.2, f.sigmoide)
    xor_couche_sortie = c.CoucheConnectee(1, 0.4, f.sigmoide)

    xor_reseau = Reseau(xor_couche_entree, xor_couche_cachee, xor_couche_cachee_2, xor_couche_sortie)

    xor_donnees = [
               [np.array([0.0, 0.0]), np.array([0.0])],
               [np.array([0.0, 1.0]), np.array([1.0])],
               [np.array([1.0, 0.0]), np.array([1.0])],
               [np.array([1.0, 1.0]), np.array([0.0])]
              ]

    xor_reseau.entrainement(xor_donnees, 50000, 0.5)
    xor_donnees_predites = xor_reseau.test(xor_donnees)

    for ((_, xor_sortie_predite), (xor_entree, xor_sortie)) in zip(xor_donnees_predites, xor_donnees):
        print(xor_entree.ravel(), '->', xor_sortie_predite.ravel(), '-->', xor_sortie.ravel())

    # Perceptron multi-couches
    sinus_couche_entree = c.CoucheEntree(1, 1)
    sinus_couche_cachee = c.CoucheConnectee(10, 0.5, f.tanh)
    sinus_couche_sortie = c.CoucheConnectee(1, 0.4, f.tanh)

    sinus_reseau = Reseau(sinus_couche_entree, sinus_couche_cachee, sinus_couche_sortie)

    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)

    sinus_donnees = [(i, j) for i, j in zip(x, y)]

    sinus_reseau.entrainement(sinus_donnees, 5000, 0.2)
    sinus_donnees_predites = sinus_reseau.test(sinus_donnees)
    sinus_sorties = [sinus_sortie for (_, sinus_sortie) in sinus_donnees_predites]

    p.plot(x, y)
    p.plot(x, sinus_sorties)
    p.show()

    # Réseau neuronal convolué
    couche_entree = c.CoucheEntree(28, 28)
    couche_convoluee = c.CoucheConvoluee(2, 0.5, f.relu, 5)
    couche_pooling = c.CouchePooling(2)
    couche_sortie = c.CoucheConnectee(10, 0.4, f.tanh)

    mnist_reseau = Reseau(couche_entree, couche_convoluee, couche_pooling, couche_sortie)

    (entrainement_images, entrainement_labels), (test_images, test_labels) = d.recuperation()
    donnees_entrainement = [(x, y) for x, y in zip(entrainement_images, entrainement_labels)]
    donnees_test = [(x, y) for x, y in zip(test_images, test_labels)]

    mnist_reseau.entrainement(donnees_entrainement[0:50], 100, 0.6)
    mnist_donnees_predites = mnist_reseau.test(donnees_test[0:10])
