import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input

K.clear_session()



data_entrenamiento = './data/train'
data_validacion = './data/validation'

"""
Parameters
"""
epocas=50
longitud, altura = 300, 300
batch_size = 32
pasos = 600
validation_steps = 120
clases = 4
lr = 0.0004


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range=0.5,
    preprocessing_function=preprocess_input,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=preprocess_input)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    directory=data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = entrenamiento_datagen.flow_from_directory(
    directory=data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


cnn = Sequential()

cnn.add(tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    pooling='avg'
))

cnn.add(Dense(clases, activation='softmax'))
cnn.layers[0].trainable = False


cnn.compile(loss="categorical_crossentropy",
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

print(entrenamiento_generador.class_indices)

with open("model/labels.txt", "w") as txt_file:
    for line in entrenamiento_generador.class_indices:
        txt_file.write(line)
        txt_file.write("\n")

cnn.fit(
    entrenamiento_generador,
    steps_per_epoch=pasos/batch_size,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps/batch_size)

print(entrenamiento_generador.class_indices)

target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/weights.h5')
