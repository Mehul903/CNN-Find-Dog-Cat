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

## Step-1: Add convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

## Step-2: Max-pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Add another convolutional layer:
## Don't need to 'input_shape' parameter for the layer because keras will understand 
## that from the previous convolutional layer.
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Step-3: Flattening
classifier.add(Flatten())


## Step-4: Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

## Compile network:
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Image preprocessing code (Thanks to keras for providing the code!!):
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                '../dataset/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                '../dataset/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

classifier.fit_generator(
            training_set,
            steps_per_epoch=8000/32,
            epochs=20,
            validation_data=test_set,
            validation_steps=2000/32)


## Making preiction on a new single observation:
import numpy as np
from keras.preprocessing import image
test_new_obs = image.load_img('../dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64, 64))
test_new_obs = image.img_to_array(test_new_obs)
test_new_obs = np.expand_dims(test_new_obs, axis = 0)

## First check the mapping of 1 and 0 to cat and/or dog.
training_set.class_indices

## Make prediction:
prediction = classifier.predict(test_new_obs)
if prediction[0][0] == 1:
    print ('Prediction is: dog')
else:
    print ('Prediction is: cat')

# =============================================================================
# Some notes about the code and CNN:
# =============================================================================


## Flow of CNN:
## Convolution --> Max Pooling --> Flattening --> Full Connection

## Step-1: Add convolutional layer
## Here (32,3,3) indicates that I'll ues 32 feature detectors and thus will have 
## 32 feature maps (Can have more if you have a GPU)!. 3x3 indicates #rows x #cols 
## in a feature detector. In other words, 3x3 is the shape of feature detector.

## (64,64,3) indicates the size of the image (in pixels) that I'll consider is 64 x 64.
## I can certainly choose 128x128 or 256x256 but I have a CPU, not GPU!

## Considered stride of 1 to go over actual features (i.e. actual image) using 
## a feature detector.


## Step-2: Max-pooling
## Will consider stride of 2 while max-pooling step. This reduces the size of a 
## feature map, i.e. we get a new feature map. Actually, the size of a new feature map 
## (or pooled feature map) is (size of original feature map/2) + 1 (because size of the 
## original feature map was 5 x 5. If the dimensiona were to be even, then size of the 
## pooled feature map will be (size of original feature  map)/2.

## One of the reasons for max-pooling is to reduce the size of feature map because it, 
## in the step of full-connection, decreases the size of neural network by decreasing 
## the number of nodes in a layer thus making it computationally more efficient.
## By reducing the size of original feature map, we don't loose performance because 
## max-pooling captures important information from original feature map.


## Why do we need max-pooled layer in a 1-D array? Why not just put all the pixel values
## of the image in a 1-D array?
## Answer: If we only put the pixel values of an image in a 1-D array then each node of
## neural network contains information about independent pixels, and does not preserve 
## spatial structure structure of an actual image. However, if we perform previous 
## steps then we are keeping spatial structure of an image and also reducing the size 
## of final neural network.
