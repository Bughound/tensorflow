import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy

K.clear_session()



data_train = './data/train'
data_validation = './data/validation'

"""
Parameters
"""
epochs=500
width, height = 300, 300
batch_size = 32
steps = 600
validation_steps = 120
filtersConv1 = 16
filtersConv2 = 32
filtersConv3 = 64
filterSize1 = (3, 3)
filterSize2 = (2, 2)
poolSize = (2, 2)
classes = 3
lr = 0.0004


##Preparamos nuestras imagenes

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range=0.5,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory=data_train,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    directory=data_validation,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')


cnn = Sequential()
cnn.add(Convolution2D(filtersConv1, filterSize1, padding ="same", input_shape=(width, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=poolSize))

cnn.add(Convolution2D(filtersConv2, filterSize2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=poolSize))

cnn.add(Convolution2D(filtersConv3, filterSize2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=poolSize))

cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

cnn.compile(loss="categorical_crossentropy",
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

print(train_generator.class_indices)

cnn.fit(
    train_generator,
    steps_per_epoch=steps/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps/batch_size)

print(train_generator.class_indices)

target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/weights.h5')
