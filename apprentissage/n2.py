from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Architecture du réseau
modele = Sequential()
# Couches de neurones
modele.add(Dense(2, input_dim=1, activation='relu'))
modele.add(Dense(1, activation='relu'))
# Couche 0
coeff = np.array([[1.,-0.5]])
biais = np.array([-1,1])
poids = [coeff,biais]
modele.layers[0].set_weights(poids)
# Couche 1
coeff = np.array([[1.0],[1.0]])
biais = np.array([0])
poids = [coeff,biais]
modele.layers[1].set_weights(poids)
# x = np.array([[1],[2],[3],[4]])
# y = np.array([[1],[4],[9],[16]])

# modele.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               
#                )
# modele.fit(x,y)

import matplotlib.pyplot as plt
liste_x =np.linspace(-2,3,num=100)

entree = np.array([[x] for x in liste_x])
sortie = modele.predict(entree)
liste_y=np.array([y[0]for y in sortie])
plt.plot(liste_x,liste_y)
plt.show()