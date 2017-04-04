# -*- coding: utf-8 -*-

import layer as ly
import network as nw

# Couches
L1 = ly.Layer("entree", 2, "sigmoid", 0.1)
L2 = ly.Layer("inter1", 5, "sigmoid", 0.3)
L3 = ly.Layer("inter2", 3, "sigmoid", 0.2)
L4 = ly.Layer("sortie", 1, "sigmoid", 0.8)

# Construction du réseau
N = nw.Network(L1.get(), L2.get(), L3.get(), L4.get())

# Données à apprendre
data = [
        [[0, 0], [0]], 
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
        ]

# Test avant l’entraînement
print("Avant :")
for d in data:
    print(d[0], " -> ", N.test(d[0]))

# Entraînement
N.train(data)

# Test après l’entraînement
print("Après :")
for d in data:
    print(d[0], " -> ", N.test(d[0]))