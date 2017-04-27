# Projet Informatique Individuel

## Contexte

Le but de ce projet est de créer une bibliothèque
qui permet de créer un réseau de neurones fonctionnel,
comprenant une ou plusieurs couches, celles-ci étant
totalement connectées ou convoluées entre elles.

Ce projet est réalisé en Python dans le cadre du
projet informatique individuel de deuxième année
à l’ENSC (École Nationale Supérieure de Cognitique).

## Installation

Pré-requis :

- [Python](https://www.python.org/downloads/)
- [Numpy](https://pypi.python.org/pypi/numpy/)

Commencez par télécharger le dépôt Github
sur votre ordinateur et décompressez-le.

Ensuite, ouvrez une invite de commandes là où vous
avez se trouve le dépôt, et tapez la commande :

```
python setup.py install
```

La bibliothèque est désormais installée ! Vous pouvez
désormais passer en Python en tapant la commande `python`
dans votre invite de commandes.

## Exemple d’utilisation

```
>>> from pii_neural_network import couche, fonction, reseau
>>> import numpy as np

>>> couche_1 = couche.CoucheEntree(2, 1)
>>> couche_2 = couche.CoucheConnectee(10, 0.5, fonction.sigmoide)
>>> couche_3 = couche.CoucheConnectee(6, 0.2, fonction.sigmoide)
>>> couche_4 = couche.CoucheConnectee(1, 0.4, fonction.sigmoide)

>>> reseau = reseau.Reseau(couche_1, couche_2, couche_3, couche_4)

>>> donnees = [
                [np.array([0.0, 0.0]), np.array([0.0])],
                [np.array([0.0, 1.0]), np.array([1.0])],
                [np.array([1.0, 0.0]), np.array([1.0])],
                [np.array([1.0, 1.0]), np.array([0.0])]
              ]

>>> reseau.entrainement(donnees, 50000, 0.5)

À l'itération 5000 l'erreur est de : 0.49932
À l'itération 10000 l'erreur est de : 0.08878
À l'itération 15000 l'erreur est de : 0.00129
À l'itération 20000 l'erreur est de : 0.00056
À l'itération 25000 l'erreur est de : 0.00035
À l'itération 30000 l'erreur est de : 0.00025
À l'itération 35000 l'erreur est de : 0.00019
À l'itération 40000 l'erreur est de : 0.00016
À l'itération 45000 l'erreur est de : 0.00013
À l'itération 50000 l'erreur est de : 0.00011

>>> donnees_predites = reseau.test(donnees)
>>> for ((_, sortie_predite), (entree, sortie)) in zip(donnees_predites, donnees):
...     print(entree.ravel(), '->', sortie_predite.ravel(), '-->', sortie.ravel())

[ 0.  0.] -> [ 0.00622684] --> [ 0.]
[ 0.  1.] -> [ 0.99169472] --> [ 1.]
[ 1.  0.] -> [ 0.99258108] --> [ 1.]
[ 1.  1.] -> [ 0.00801979] --> [ 0.]
```
