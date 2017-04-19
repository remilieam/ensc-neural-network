# -*- coding: utf-8 -*-

import couche as c
import numpy as np
import fonction as f
import donnees as d

class Reseau:
    def __init__(self, *couches):
        assert len(couches) > 0
        assert isinstance(couches[0], c.CoucheEntree)
        self.couche_entree = couches[0]
        assert isinstance(couches[-1], c.CoucheConnectee)
        self.couche_sortie = couches[-1]
        self.couches = [(couche_precedente, couche) for couche_precedente, couche in zip(couches[:-1], couches[1:])]
        for couche_precedente, couche in self.couches:
            couche.connexion(couche_precedente)

    def prediction(self, entrees):
        self.couche_entree.neurones = entrees
        for couche_precedente, couche in self.couches:
            couche.prediction(couche_precedente)

    def propagation(self, donnees, coef_apprentissage):
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
        for i in range(nb_entrainements):
            erreur = self.propagation(donnees, coef_apprentissage)
            if ((i+1) % (nb_entrainements/10)) == 0 :
                    print("À l’itération", (i+1), "l’erreur est de : %-.5f" %erreur)

    def test(self, donnees):
        for entrees, sorties in donnees:
            self.prediction(entrees)
            print(entrees.ravel(), '->', self.couche_sortie.neurones.ravel(), '-->', sorties.ravel())


if __name__ == "__main__":
    # Perceptron multi-couches
    couche_entree = c.CoucheEntree(2, 1)
    couche_cachee = c.CoucheConnectee(10, 0.5, f.sigmoide)
    couche_cachee_2 = c.CoucheConnectee(6, 0.2, f.sigmoide)
    couche_sortie = c.CoucheConnectee(1, 0.4, f.sigmoide)

    reseau = Reseau(couche_entree, couche_cachee, couche_cachee_2, couche_sortie)

    donnees = [
               [np.array([0.0, 0.0]), np.array([0.0])],
               [np.array([0.0, 1.0]), np.array([1.0])],
               [np.array([1.0, 0.0]), np.array([1.0])],
               [np.array([1.0, 1.0]), np.array([0.0])]
              ]

    reseau.entrainement(donnees, 50000, 0.5)
    reseau.test(donnees)

    # Réseau neuronal convolué
    couche_entree = c.CoucheEntree(28, 28)
    couche_convoluee = c.CoucheConvoluee(2, 0.5, f.sigmoide, 5)
    couche_pooling = c.CouchePooling(2)
    couche_sortie = c.CoucheConnectee(10, 0.4, f.sigmoide)

    reseau_mnist = Reseau(couche_entree, couche_convoluee, couche_pooling, couche_sortie)

    donnees_mnist = d.Donnees('MNIST_data')
    (entrainement_images, entrainement_labels), (test_images, test_labels) = donnees_mnist.recuperation()
    donnees_entrainement = [(x, y) for x, y in zip(entrainement_images, entrainement_labels)]
    donnees_test = [(x, y) for x, y in zip(test_images, test_labels)]

    reseau_mnist.entrainement(donnees_entrainement[0:1000], 10, 0.1)
    reseau_mnist.test(donnees_test[0:2])
