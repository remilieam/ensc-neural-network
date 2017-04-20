#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes pour créer les couches d’un réseau de
neurones
"""

import pii_neural_network.fonction as f
import numpy as np
__all__ = ['CoucheEntree', 'CoucheConnectee',
           'CoucheConvoluee', 'CouchePooling']


class Couche:
    """
    Classe abstraite
    """
    def __init__(self):
        """
        Constructeur => tous les attributs sont vides par défaut
        """
        self.profondeur = None
        self.hauteur = None
        self.largeur = None
        self.nb_neurones = None
        self.poids = None
        self.biais = None
        self.neurones = None
        self.initialisation = None
        self.activation = None
        self.derivee_activation = None

    def connexion(self, couche_precedente):
        """
        Méthode abstraite
        """
        raise AssertionError

    def prediction(self, couche_precedente):
        """
        Méthode abstraite
        """
        raise AssertionError

    def propagation(self, couche_precedente, delta):
        """
        Méthode abstraite
        """
        raise AssertionError


class CoucheEntree(Couche):
    """
    Classe pour la première couche de neurones d’un réseau de neurones
    """
    # Couche d’entrée d’une profondeur de 1, d’une hauteur donnée
    # et d’une largeur donnée
    def __init__(self, hauteur, largeur):
        """
        Constructeur
        
        Arguments :
        -----------
        
        hauteur : hauteur de la couche
        largeur : largeur de la couche
        
        Exemple :
        ---------
        
        Pour avoir une première couche ayant 24 neurones de haut et 12 de large :
        
        >>> couche = CoucheEntree(24, 12)
        """
        Couche.__init__(self)
        self.profondeur = 1
        self.hauteur = hauteur
        self.largeur = largeur
        self.nb_neurones = self.profondeur * self.hauteur * self.largeur
        self.derivee_activation = lambda x: x

    # Pas possible de connecter cette couche à une précédente couche
    # vu que c’est la première
    def connexion(self, couche_precedente):
        """
        Méthode non-définie car une couche d’entrée n’a pas de 
        couche précédente
        """
        raise AssertionError
    
    # Pas possible de calculer les valeurs des neurones de cette couche 
    # en fonction des valeurs des neurones de la précédente couche car
    # cette couche est la première
    def prediction(self, couche_precedente):
        """
        Méthode non-définie car une couche d’entrée n’a pas de 
        couche précédente
        """
        raise AssertionError

    # Pas possible de calculer les variations à effectuer dans la matrice des
    # poids ou dans la matrice de biais, ni à calculer l’erreur pour la couche
    # précédente parce que cette couche est la première
    def propagation(self, couche_precedente, delta):
        """
        Méthode non-définie car une couche d’entrée n’a pas de 
        couche précédente
        """
        raise AssertionError

class CoucheConnectee(Couche):
    """
    Classe pour les couches entièrement connectées d’un réseau de neurones
    """
    # Couche entièrement connectée d’une profondeur de 1, d’une hauteur donnée
    # et d’une largeur de 1
    def __init__(self, hauteur, initialisation, activation):
        """
        Constructeur
        
        Arguments :
        -----------
        
        hauteur        : hauteur de la couche de neurones
        initialisation : amplitude de l’intervalle dans lequel seront
                         initialisés les poids synaptiques
        activation     : fonction d’activation
        
        Exemple :
        ---------
        
        Pour avoir une couche entièrement connectée ayant 5 neurones,
        une valeur d’initialisation de 0.2 et la fonction tanh pour
        fonction d’activation :
        
        >>> couche = CoucheConnectee(5, 0.2, f.sigmoide)
        """
        Couche.__init__(self)
        self.profondeur = 1
        self.hauteur = hauteur
        self.largeur = 1
        self.nb_neurones = self.profondeur * self.hauteur * self.largeur
        self.initialisation = initialisation
        self.activation = activation
        self.derivee_activation = getattr(f, "derivee_%s" % activation.__name__)

    # En fonction de la dimension de la précédente couche, construction
    # la matrice des poids dont le nombre de lignes vaut le nombre de neurones
    # de notre couche et le nombre de colonnes le nombre de neurones de la 
    # précédente couche, et la matrice de biais (matrice colonne dont le 
    # nombre de lignes correspond au nombre de neurones de notre couche)
    def connexion(self, couche_precedente):
        """
        Connecte notre couche une autre couche, généralement la couche
        qui la précède dans le réseau, en créant la matrice des poids
        synaptiques et la matrice des biais entre les 2 couches
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        
        Exemple :
        ---------
        
        Pour connecter notre couche couche_2 à la couche couche_1 :
        
        >>> couche_2.connexion(couche_1)
        """
        self.poids = np.random.uniform(-self.initialisation, self.initialisation, (self.nb_neurones, couche_precedente.nb_neurones))
        self.biais = np.ones((self.nb_neurones, 1))

    # En fonction des valeurs de sortie des neurones de la couche précédente,
    # calcul les valeurs de sortie des neurones de notre couche
    def prediction(self, couche_precedente):
        """
        Calcule les valeurs des neurones de notre couche en fonction des
        valeurs des neurones de la couche précédente dans le réseau
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        """
        # La couche précédente devient une matrice colonne
        neurones_precedents = couche_precedente.neurones.reshape((couche_precedente.neurones.size, 1))

        # Calcul des valeurs des neurones de notre couche juste avant leur
        # entrée (multiplication matricielle
        # des poids et des valeurs des neurones de la couche précédente +
        # ajout du biais)
        self.neurones = self.activation(np.dot(self.poids, neurones_precedents) + self.biais)

    # En fonction des valeurs de sortie des neurones de la précédente couche
    # et de la valeur de l’erreur précédemment calculée pour notre couche,
    # calcul des variations des poids et du biais, 
    # ainsi que de la nouvelle erreur calculée
    def propagation(self, couche_precedente, delta):
        """
        Calcule les variations des poids synaptiques et des biais à effectuer
        entre notre couche à la couche précédente afin de minimiser l’erreur
        quadratique (dérivée de l’erreur par rapport aux poids / aux biais)
        
        Arguments :
        -----------
        
        couche_precedente : instance de la classe Couche
        delta             : erreur
        
        Retours :
        ---------
        
        nouveaux_poids  : variations à effectuer sur les poids synaptique
        nouveaux_biais  : variations à effectuer sur les biais
        delta_precedent : erreur à propager pour la précédente couche
        """
        # Vérification que la taille est cohérente
        assert delta.shape == self.neurones.shape

        # La couche précédente devient une matrice colonne
        neurones_precedents = couche_precedente.neurones.reshape((couche_precedente.neurones.size, 1))

        # Les variations des poids sont calculées en faisant le produit 
        # matriciel de l’erreur de notre couche et de la transposée de la matrice
        # contenant les valeurs des neurones de la couche précédente
        nouveaux_poids = np.dot(delta, neurones_precedents.T)

        # La variation du biais correspond à l’erreur de notre couche
        nouveaux_biais = np.copy(delta)

        # L’erreur pour la précédente couche est calculée et faisant le produit 
        # matricielle de la transposée de la maitrice des poids par l’erreur
        # actuelle puis en multipliant un à un chaque valeur par la valeur des
        # neurones de la couche précédente dérivée
        delta_precedent = np.dot(self.poids.T, delta).reshape(couche_precedente.neurones.shape)*couche_precedente.derivee_activation(couche_precedente.neurones)

        # Renvoi des variations à effectuer et de l’erreur pour la couche précédente
        return nouveaux_poids, nouveaux_biais, delta_precedent

        
class CoucheConvoluee(Couche):
    """
    Classe pour les couches de convolution d’un réseau de neurones
    """
    # Couche convoluée de profondeur donnée et de taille de filtre donné et
    # de pas 1 par défaut
    def __init__(self, profondeur, initialisation, activation, taille_filtre, pas = 1):
        """
        Constructeur
        
        Arguments :
        -----------
        
        profondeur     : profondeur de la couche de neurones
        initialisation : amplitude de l’intervalle dans lequel seront
                         initialisés les poids synaptiques
        activation     : fonction d’activation
        taille_filtre  : taille du filtre qui permettra la convolution
        pas            : chevauchement entre les différentes surfaces de
                         traitement, à 1 par défaut
        
        Exemple :
        ---------
        
        Pour avoir une couche de convolution de profondeur 2, avec
        une valeur d’initialisation de 0.4 et la fonction ReLu pour
        fonction d’activation, un filtre de 5 par 5 et un pas de 1 :
        
        >>> couche = CoucheConnectee(2, 0.4, f.relu, 5)
        
        ou
        
        >>> couche = CoucheConnectee(2, 0.4, f.relu, 5, 1)
        """
        Couche.__init__(self)
        self.profondeur = profondeur
        self.taille_filtre = taille_filtre
        self.pas = 1
        self.initialisation = initialisation
        self.activation = activation
        self.derivee_activation = getattr(f, "derivee_%s" % activation.__name__)

    # Connexion de notre couche à la couche précédente via la matrice de poids
    # Récupération également de la taille de la couche et de la matrice de biais
    def connexion(self, couche_precedente):
        """
        Connecte notre couche une autre couche, généralement la couche
        qui la précède dans le réseau, en créant la matrice des poids
        synaptiques et la matrice des biais entre les 2 couches
        
        Permet également, en fonction de la couche précédente, de déduire
        les dimensions de notre couche
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        
        Exemple :
        ---------
        
        Pour connecter notre couche couche_2 à la couche couche_1 :
        
        >>> couche_2.connexion(couche_1)
        """
        # Déduction de la taille de la couche grâce à la taille du filtre utilisé
        self.hauteur = ((couche_precedente.hauteur - self.taille_filtre)//self.pas) + 1
        self.largeur  = ((couche_precedente.largeur  - self.taille_filtre)//self.pas) + 1
        self.nb_neurones = self.profondeur * self.hauteur * self.largeur

        # Calcul de la matrice des poids et la matrice du biais
        self.poids = np.random.uniform(-self.initialisation, self.initialisation, (self.profondeur, couche_precedente.profondeur, self.taille_filtre, self.taille_filtre))
        self.biais = np.zeros((self.profondeur, 1))

    # Calcul des valeurs des neurones de sortie de notre couche en fonction
    # des valeurs des neurones de sortie de la couche précédente
    def prediction(self, couche_precedente):
        """
        Calcule les valeurs des neurones de notre couche en fonction des
        valeurs des neurones de la couche précédente dans le réseau
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        """
        # Vérification que la couche précédente est compatible avec le réseau
        assert self.poids.shape == (self.profondeur, couche_precedente.profondeur, self.taille_filtre, self.taille_filtre)
        assert self.biais.shape == (self.profondeur, 1)
        assert couche_precedente.neurones.ndim == 3

        # Récupération des valeurs de sorties de la couche précédente
        neurones_precedents = couche_precedente.neurones
        hauteur_precedente = couche_precedente.hauteur
        largeur_precedente = couche_precedente.largeur

        # Initialisation de la matrice contenant les valeurs des neurones
        # de notre couche
        self.neurones = np.zeros((self.profondeur, self.hauteur, self.largeur))

        # Calcul des valeurs des neurones de notre couche
        # Parcours des couches de sortie
        for r in range(self.profondeur):
            # Parcours des couches d’entrée
            for t in range(couche_precedente.profondeur):
                # Récupération du filtre
                filtre = self.poids[r, t]
                # Parcours de la hauteur avec un pas de 1 (donc i = m)
                for i, m in enumerate(range(0, hauteur_precedente - self.taille_filtre + 1, self.pas)):
                    # Parcours de la largeur avec un pas de 1 (donc j = n)
                    for j, n in enumerate(range(0, largeur_precedente - self.taille_filtre + 1, self.pas)):
                        # Récupération de la matrice des valeurs des neurones
                        # de la couche précédente et dont la taille est celle
                        # du filtre
                        neurones_precedents_filtre = neurones_precedents[t, m:m + self.taille_filtre, n:n + self.taille_filtre]
                        # Calcul de la valeur de sortie (somme des produits des
                        # valeurs des neurones de la couche de sortie par
                        # les valeurs du filtre)
                        self.neurones[r, i, j] += np.correlate(neurones_precedents_filtre.ravel(), filtre.ravel())

        # Ajout du biais
        for r in range(self.profondeur):
            self.neurones[r] += self.biais[r]

        # Passage dans la fonction d’activation
        self.neurones = np.vectorize(self.activation)(self.neurones)

    # En fonction des valeurs de sortie des neurones de la précédente couche
    # et de la valeur de l’erreur précédemment calculée pour notre couche,
    # calcul des variations des poids et du biais, 
    # ainsi que de la nouvelle erreur calculée
    def propagation(self, couche_precedente, delta):
        """
        Calcule les variations des poids synaptiques et des biais à effectuer
        entre notre couche à la couche précédente afin de minimiser l’erreur
        quadratique (dérivée de l’erreur par rapport aux poids / aux biais)
        
        Arguments :
        -----------
        
        couche_precedente : instance de la classe Couche
        delta             : erreur
        
        Retours :
        ---------
        
        nouveaux_poids  : variations à effectuer sur les poids synaptique
        nouveaux_biais  : variations à effectuer sur les biais
        delta_precedent : erreur à propager pour la précédente couche
        """
        assert delta.shape[0] == self.profondeur

        # Récupération des valeurs de sorties de la couche précédente
        neurones_precedents = couche_precedente.neurones

        # Calcul des variations de poids à effectuer sur notre couche
        # Initialisation de la matrice des variations de poids
        nouveaux_poids = np.empty_like(self.poids)
        # Parcours des couches de sortie
        for r in range(self.profondeur):
            # Parcours des couches d’entrée
            for t in range(couche_precedente.profondeur):
                # Parcours du filtre en hauteur
                for h in range(self.taille_filtre):
                    # Parcours du filtre en largeur
                    for v in range(self.taille_filtre):
                        # Récupération de la matrice des valeurs des neurones
                        # de la couche précédente et dont la taille est celle
                        # du filtre
                        neurones_precedents_filtre = neurones_precedents[t, 
                                                                         v:v + self.hauteur - self.taille_filtre + 1:self.pas,
                                                                         h:h + self.largeur - self.taille_filtre + 1:self.pas
                                                                        ]
                        # Récupération de la matrice des valeurs des erreurs
                        # calculée précédemment et dont la taille est celle
                        # du filtre
                        delta_filtre  =  delta[r,
                                               v:v + self.hauteur - self.taille_filtre + 1:self.pas,
                                               h:h + self.largeur - self.taille_filtre + 1:self.pas
                                              ]
                        assert neurones_precedents_filtre.shape == delta_filtre.shape
                        # Calcul de la modification de poids à effectuer
                        nouveaux_poids[r, t, h, v] = np.sum(neurones_precedents_filtre*delta_filtre)

        # Récupération des variations à appliquer sur le biais
        # Initialisation de la matrice de variations de biais
        nouveaux_biais = np.empty((self.profondeur, 1))
        # Parcours des chouches de sortie
        for r in range(self.profondeur):
            # Récupération de la variation de biais pour la couche r
            nouveaux_biais[r] = np.sum(delta[r])

        # Calcul de l’erreur pour la précédente couche
        delta_precedent = np.zeros_like(neurones_precedents)
        # Parcours des couches de sortie
        for r in range(self.profondeur):
            # Parcours des couches d’entrée
            for t in range(couche_precedente.profondeur):
                # Récupération du filtre
                filtre = self.poids[r, t]
                # Parcours du filtre dans la hauteur avec un pas de 1
                for i, m in enumerate(range(0, couche_precedente.hauteur - self.taille_filtre + 1, self.pas)):
                    # Parcours du filtre dans la largeur avec un pas de 1
                    for j, n in enumerate(range(0, couche_precedente.largeur - self.taille_filtre + 1, self.pas)):
                        # Calcul de l’erreur (1° partie)
                        delta_precedent[t, m:m + self.taille_filtre, n:n + self.taille_filtre] += filtre*delta[r, i, j]
        # Calcul de l’erreur (2° partie)
        delta_precedent *= couche_precedente.derivee_activation(couche_precedente.neurones)

        # Renvoi des variations à effectuer et de l’erreur pour la précédente
        # couche
        return nouveaux_poids, nouveaux_biais, delta_precedent


class CouchePooling(Couche):
    """
    Classe pour les couches de pooling d’un réseau de neurones
    Une couche de pooling est toujours précédée par une couche de convolution
    """
    def __init__(self, taille_pool):
        """
        Constructeur
        
        Argument :
        ----------
        
        taille_pool : taille du regroupement, généralement de 2
        
        Exemple :
        ---------
        
        Pour avoir une couche de pooling qui permet de regrouper une surface
        de 2 par 2 en une surface unique
        
        >>> couche = CoucheConnectee(2)
        """
        Couche.__init__(self)
        self.taille_pool = taille_pool
        self.derivee_activation = lambda x: x

    def connexion(self, couche_precedente):
        """
        Connecte notre couche une autre couche, généralement la couche
        qui la précède dans le réseau
        
        Permet également, en fonction de la couche précédente, de déduire
        les dimensions de notre couche
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        
        Exemple :
        ---------
        
        Pour connecter notre couche couche_2 à la couche couche_1 :
        
        >>> couche_2.connexion(couche_1)
        """
        # Vérification que la couche précédente est bien une couche convoluée
        assert isinstance(couche_precedente, CoucheConvoluee)
        
        # Récupération des dimensions de notre couche en fonction de la taille
        # du pooling
        self.profondeur = couche_precedente.profondeur
        self.hauteur = ((couche_precedente.hauteur - self.taille_pool)//self.taille_pool) + 1
        self.largeur = ((couche_precedente.largeur - self.taille_pool)//self.taille_pool) + 1
        self.nb_neurones = self.profondeur * self.hauteur * self.largeur

        # Pas de matrice de poids ni de biais entre la couche convoluée et 
        # notre couche
        self.poids = np.empty((0))
        self.biais = np.empty((0))

    def prediction(self, couche_precedente):
        """
        Calcule les valeurs des neurones de notre couche en fonction des
        valeurs des neurones de la couche précédente dans le réseau
        
        Argument :
        ----------
        
        couche_precedente : instance de la classe Couche
        """
        # Vérification de la couche précédente est compatible avec notre couche
        assert self.poids.size == 0
        assert self.biais.size == 0
        assert isinstance(couche_precedente, CoucheConvoluee)
        assert couche_precedente.profondeur == self.profondeur
        assert couche_precedente.hauteur % self.taille_pool == 0
        assert couche_precedente.neurones.ndim == 3

        neurones_precedents = couche_precedente.neurones

        # Initialisation des valeurs des neurones de notre couche
        self.neurones = np.zeros((self.profondeur, self.hauteur, self.largeur))
        # Parcours en profondeur des 2 couches (qui ont normalement la même
        # profondeur)
        for r, t in zip(range(self.profondeur), range(couche_precedente.profondeur)):
            assert r == t
            # Parcours de la couche dans sa hauteur avec un pas égal à la taille
            # du pooling
            for i, m in enumerate(range(0, couche_precedente.hauteur, self.taille_pool)):
                # Parcours de la couche dans sa largeur avec un pas égal à la 
                # taille du pooling
                for j, n in enumerate(range(0, couche_precedente.largeur, self.taille_pool)):
                    # Récupération de la matrice des valeurs des neurones de la
                    # couche précédente et dont la taille est celle
                    # du filtre
                    neurones_precedents_pool = neurones_precedents[t, m:m + self.taille_pool, n:n + self.taille_pool]
                    assert neurones_precedents_pool.shape == (self.taille_pool, self.taille_pool)
                    # Le neurone prend la valeur maximale parmi les valeurs
                    # des neurones de neurones_precedents_pool
                    self.neurones[r, i, j] = np.max(neurones_precedents_pool)

    def propagation(self, couche_precedente, delta):
        """
        Calcule l’erreur à propager pour la précédente couche
        
        Arguments :
        -----------
        
        couche_precedente : instance de la classe Couche
        delta             : erreur
        
        Retours :
        ---------
        
        nouveaux_poids  : Ø
        nouveaux_biais  : Ø
        delta_precedent : erreur à propager pour la précédente couche
        """
        # Vérification de la couche précédente est compatible avec notre couche
        assert self.poids.size == 0
        assert self.biais.size == 0
        assert isinstance(couche_precedente, CoucheConvoluee)
        assert couche_precedente.profondeur == self.profondeur
        assert couche_precedente.neurones.ndim == 3
        assert delta.shape == (self.profondeur, self.hauteur, self.largeur)

        neurones_precedents = couche_precedente.neurones

        nouveaux_poids = np.array([])
        nouveaux_biais = np.array([])

        delta_precedent = np.empty_like(neurones_precedents)
        # Parcours en profondeur des 2 couches (qui ont normalement la même
        # profondeur)
        for r, t in zip(range(self.profondeur), range(couche_precedente.profondeur)):
            assert r == t
            # Parcours de la couche dans sa hauteur avec un pas égal à la taille
            # du pooling
            for i, m in enumerate(range(0, couche_precedente.hauteur, self.taille_pool)):
                # Parcours de la couche dans sa largeur avec un pas égal à la 
                # taille du pooling
                for j, n in enumerate(range(0, couche_precedente.largeur, self.taille_pool)):
                    # Récupération de la matrice des valeurs des neurones de la
                    # couche précédente et dont la taille est celle
                    # du filtre
                    neurones_precedents_pool = neurones_precedents[t, m:m + self.taille_pool, n:n + self.taille_pool]
                    assert neurones_precedents_pool.shape == (self.taille_pool, self.taille_pool)
                    # Récupération du numéro du neurone dont la valeur est
                    # maximale
                    numero_max = neurones_precedents_pool.argmax()
                    # Récupération du numéro de ligne et de colonne du neurone
                    # dont la valeur est maximale
                    position_max = np.unravel_index(numero_max, neurones_precedents_pool.shape)
                    # Initialisation de l’erreur
                    delta_precedent_pool = np.zeros_like(neurones_precedents_pool)
                    # Le coefficient correspondant au neurone le plus fort
                    # prend toute l’erreur et les autres rien
                    delta_precedent_pool[position_max] = delta[t, i, j]
                    # Modification sur l’erreur globale
                    delta_precedent[r, m:m + self.taille_pool, n:n + self.taille_pool] = delta_precedent_pool

        return nouveaux_poids, nouveaux_biais, delta_precedent
