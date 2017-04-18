# -*- coding: utf-8 -*-

import fonction as f
import numpy as np

# Classe abstraite
class Couche:
    def __init__(self):
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
        raise AssertionError

    def prediction(self, couche_precedente):
        raise AssertionError

    def propagation(self, couche_precedente, delta):
        raise AssertionError


class CoucheEntree(Couche):
    # Couche d’entrée d’une profondeur de 1, d’une hauteur donnée
    # et d’une largeur donnée
    def __init__(self, hauteur, largeur):
        Couche.__init__(self)
        self.profondeur = 1
        self.hauteur = hauteur
        self.largeur = largeur
        self.nb_neurones = self.profondeur * self.hauteur * self.largeur
        self.derivee_activation = lambda x: x

    # Pas possible de connecter cette couche à une précédente couche
    # vu que c’est la première
    def connexion(self, couche_precedente):
        raise AssertionError
    
    # Pas possible de calculer les valeurs des neurones de cette couche 
    # en fonction des valeurs des neurones de la précédente couche car
    # cette couche est la première
    def prediction(self, couche_precedente):
        raise AssertionError

    # Pas possible de calculer les variations à effectuer dans la matrice des
    # poids ou dans la matrice de biais, ni à calculer l’erreur pour la couche
    # précédente parce que cette couche est la première
    def propagation(self, couche_precedente, delta):
        raise AssertionError

class CoucheConnectee(Couche):
    # Couche entièrement connectée d’une profondeur de 1, d’une hauteur donnée
    # et d’une largeur de 1
    def __init__(self, hauteur, initialisation, activation):
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
        self.poids = np.random.uniform(-self.initialisation, self.initialisation, (self.nb_neurones, couche_precedente.nb_neurones))
        self.biais = np.ones((self.nb_neurones, 1))

    # En fonction des valeurs de sortie des neurones de la couche précédente,
    # calcul les valeurs de sortie des neurones de notre couche
    def prediction(self, couche_precedente):
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
    # Couche convoluée de profondeur donnée et de taille de filtre donné et
    # de pas 1 par défaut
    def __init__(self, profondeur, initialisation, activation, taille_filtre, pas = 1):
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
    def __init__(self, taille_pool):
        Couche.__init__(self)
        self.taille_pool = taille_pool
        self.derivee_activation = lambda x: x

    def connexion(self, couche_precedente):
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
