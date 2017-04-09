# -*- coding: utf-8 -*-

import numpy as np
import layer as ly
import network as nw

# Couches
L1 = ly.Layer("entree", 2, "sigmoid", 0.5)
L2 = ly.Layer("inter1", 4, "sigmoid", 0.2)
L3 = ly.Layer("inter2", 5, "sigmoid", 0.3)
L4 = ly.Layer("inter3", 3, "sigmoid", 0.7)
L5 = ly.Layer("sortie", 1, "sigmoid", 0.8)

# Construction du réseau
N = nw.Network(L1.get(), L2.get(), L3.get(), L4.get(), L5.get())

# Récupération des données
tableau = np.loadtxt('data.txt').reshape(3000, 3)

# Création des entrées
inputs = np.ones((3000, 2))
compteur = 0
for i in range(0, 3000, 2):
    inputs[i] = np.delete(tableau[compteur], 0)
    inputs[i] = inputs[i]/800.0
    inputs[i+1] = np.delete(tableau[compteur+1500], 0)
    inputs[i+1] = inputs[i+1]/800.0
    compteur += 1
    
# Création des sorties visées
targets = np.ones((3000, 1))
for i in range(3000):
    if i%2 == 0:
        targets[i] = 0.1
    else:
        targets[i] = 0.9
    
## Données à apprendre
#inputs = np.array([
#        [0, 0], 
#        [0, 1],
#        [1, 0],
#        [1, 1]
#        ])
#
#targets = np.array([
#        [1, 1], 
#        [1, 0],
#        [0, 1],
#        [0, 0]
#        ])

## Test avant l’entraînement
print("Avant :")
p=0
m=0
for i in inputs:
    if N.test(i) > 0.5:
        p+=1
    else:
        m+=1
print(p, m)

# Entraînement
N.train(inputs, targets, 100, 0.8)

# Test après l’entraînement
print("Après :")
p=0
m=0
for i in inputs:
    if N.test(i) > 0.5:
        p+=1
    else:
        m+=1
print(p, m)