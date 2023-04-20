#!/usr/bin/env python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
        [
            keras.Input(10, name="first"),
            layers.Dense(2, activation='relu', name="second"),
            layers.Dense(3, activation='relu', name="third"),
            layers.Dense(4, activation='relu', name="fourth"),
            layers.Dense(2, activation='relu', name="fifth"),
        ]
    )

for layer in model.layers:
    print(layer.name, layer.get_weights())

weights = model.get_weights()
for i, w in enumerate(weights):
    print(i, w.shape)

# print("weight", weights)
# print("set", model.set_weights(weights))
model.save("test.h5")

for layer in model.layers[:-2]:
    layer.trainable = False
