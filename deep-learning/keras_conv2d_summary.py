from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid',
                 activation='relu', input_shape=(200, 200, 1)))
model.summary()

# Formula: Number of Parameters in a Convolutional Layer
# K - the number of filters in the convolutional layer
# F - the height and width of the convolutional filters
# D_in - the depth of the previous layer