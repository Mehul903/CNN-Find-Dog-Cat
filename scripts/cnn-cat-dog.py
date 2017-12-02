## Import packages for building a CNN:
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


## Initialize CNN:
classifier = Sequential()


## Flow of CNN:
## Convolution --> Max Pooling --> Flattening --> Full Connection
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3)), activation = 'relu')
