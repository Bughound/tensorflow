import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

width, height = 300, 300
model = './model/model.h5'
weights = './model/weights.h5'
cnn = load_model(model)
cnn.load_weights(weights)

def predict(file):
  x = load_img(file, target_size=(width, height))
  x = img_to_array(x) / 255
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)

  print(100 * np.max(tf.nn.softmax(array[0])))
  print(result)

  return answer

# predict('imagenes/arañas/ia_100000002.jpeg')