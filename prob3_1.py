# -*- coding: utf-8 -*-
"""dl-hw-prob3-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BQoTLjAwYIHn7-o_UM2e7brEjf0YUBxD
"""

import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D

"""
!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
!tar -xzf cifar-10-python.tar.gz
!ls
"""
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(11, 11), padding="same", activation="relu",
    data_format="channels_last", input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",
    data_format="channels_last"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",
    data_format="channels_last"))
model.add(AveragePooling2D(pool_size=(16, 16), data_format="channels_last"))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.8),
              metrics=['accuracy'])

def unpickle_cifar10(num):
    if num > 0:
        with open("cifar-10-batches-py/data_batch_" + str(num), 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
    else:
        with open("cifar-10-batches-py/test_batch", 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
    images = d[b'data']
    labels = d[b'labels']
    images = images.astype("float32") / 255.0
    images = (images - 0.4747) / 0.2525
    return images.reshape((-1, 32, 32, 3)), labels

images, labels = unpickle_cifar10(1)
for i in [2,3,4,5]:
    img, lbl = unpickle_cifar10(i)
    images = np.vstack((images, img))
    labels = labels + lbl

model.fit(images, labels, epochs=100, batch_size=100)

test_images, test_labels = unpickle_cifar10(0)
model.evaluate(test_images, test_labels, batch_size=100)

weights, biases = model.layers[0].get_weights()
weights.shape

import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=8, ncols=8, sharey=True, sharex=True)
for i in range(8):
    for j in range(8):
        curr_weight_0 = weights[:, :, 0, i*8+j].reshape((11, 11, 1))
        curr_weight_0 = curr_weight_0 - np.amin(curr_weight_0)
        curr_weight_0 = curr_weight_0 / np.amax(curr_weight_0)
        curr_weight_1 = weights[:, :, 1, i*8+j].reshape((11, 11, 1))
        curr_weight_1 = curr_weight_1 - np.amin(curr_weight_1)
        curr_weight_1 = curr_weight_1 / np.amax(curr_weight_1)
        curr_weight_2 = weights[:, :, 2, i*8+j].reshape((11, 11, 1))
        curr_weight_2 = curr_weight_2 - np.amin(curr_weight_2)
        curr_weight_2 = curr_weight_2 / np.amax(curr_weight_2)
        axes[i, j].imshow(np.dstack((curr_weight_0, curr_weight_1, curr_weight_2)))
plt.show()

