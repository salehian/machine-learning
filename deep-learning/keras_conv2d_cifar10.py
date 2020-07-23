import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10

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
