import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras.utils import np_utils


# Load the pre-shuffled train and test data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Plot the first 18 images
fig = plt.figure(figsize=(20,5))
for i in range(18):
  ax = fig.add_subplot(3, 6, i+1)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(np.squeeze(X_train[i]))

# Rescale
X_train = X_train.astype('float32')/255
y_train = y_train.astype('float32')/255

# One-hot encode the labels
num_classes = len(unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Break a part from training set for validation set
(X_train, X_valid) = X_train[9000:], X_train[:9000]
(y_train, y_valid) = y_train[9000:], y_train[:9000]
