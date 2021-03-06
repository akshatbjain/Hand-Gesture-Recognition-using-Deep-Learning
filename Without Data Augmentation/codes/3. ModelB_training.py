from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

batch_size = 20
nClasses = 5
dataAugmentation = False

#################################################
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/training',
        target_size=(224, 224),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        '/home/akshat/deep_learning/Consolidated_Data_Set/Dataset/validation',
        target_size=(224, 224),
        batch_size=batch_size)
#################################################

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same',input_shape=(3,224,224)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3 ))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64,3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)

# serialize model to JSON
model_json = model.to_json()
with open("ModelB.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("ModelB.h5")
print("Saved model to disk")

