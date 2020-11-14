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



data_train = './data/train'
data_validation = './data/validation'

"""
Parameters
"""
epochs=50
width, height = 300, 300
batch_size = 32
steps = 600
validation_steps = 120
classes = 4
lr = 0.0004


##Images preparation

train_datagen = ImageDataGenerator(
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

train_generator = train_datagen.flow_from_directory(
    directory=data_train,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    directory=data_validation,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')


cnn = Sequential()

cnn.add(tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    pooling='avg'
))

cnn.add(Dense(classes, activation='softmax'))
cnn.layers[0].trainable = False


cnn.compile(loss="categorical_crossentropy",
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

print(train_generator.class_indices)

with open("model/labels.txt", "w") as txt_file:
    for line in train_generator.class_indices:
        txt_file.write(line)
        txt_file.write("\n")

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
