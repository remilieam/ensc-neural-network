# -*- coding: utf-8 -*-

import numpy as np
import layer as ly
import network as nw

# Couches
L1 = ly.Layer("entree", 2, "sigmoid", 0.5)
L2 = ly.Layer("inter1", 4, "sigmoid", 0.2)
L3 = ly.Layer("inter2", 5, "sigmoid", 0.3)
L4 = ly.Layer("inter3", 3, "sigmoid", 0.7)
L5 = ly.Layer("sortie", 2, "sigmoid", 0.8)

# Construction du réseau
N = nw.Network(L1.get(), L2.get(), L3.get(), L4.get(), L5.get())

# Données à apprendre
data = [
        [[0, 0], [1, 1]], 
        [[0, 1], [1, 0]],
        [[1, 0], [0, 1]],
        [[1, 1], [0, 0]]
        ]

# Test avant l’entraînement
print("Avant :")
for d in data:
    print(d[0], "->", N.test(np.array(d[0]))[0])

# Entraînement
N.train(data, 1000, 0.2)

# Test après l’entraînement
print("Après :")
for d in data:
    print(d[0], "->", N.test(np.array(d[0]))[0])