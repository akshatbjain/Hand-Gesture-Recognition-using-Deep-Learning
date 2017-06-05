from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

nClasses = 5
batch_size = 50
epochs = 50

#################################################

train_datagen = ImageDataGenerator(rotation_range=360.,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/media/akshat/Akshat/Linux Backup/deep_learning/Consolidated_Data_Set/Dataset/training',
        target_size=(224, 224),
        batch_size=batch_size)

validation_generator = validation_datagen.flow_from_directory(
        '/media/akshat/Akshat/Linux Backup/deep_learning/Consolidated_Data_Set/Dataset/validation',
        target_size=(224, 224),
        batch_size=batch_size)

#################################################

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 7, 7, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))
model.summary()

#############################m####################

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit_generator(train_generator,
        samples_per_epoch=2000,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=800)

# serialize model to JSON
model_json = model.to_json()
with open("ModelA1.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("ModelA1.h5")
print("Saved model to disk")
