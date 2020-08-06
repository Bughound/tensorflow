import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 160, 160
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

img = load_img('perro.jpg', target_size=(longitud, altura))

img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0) # Create a batch

predictions = model.predict(img_array)

tf.keras.applications.mobilenet_v2.decode_predictions(predictions)


print(predictions)
score = tf.nn.softmax(predictions[0])

print(100 * np.max(score))
class_names = ['Perro', 'Gato']

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
 