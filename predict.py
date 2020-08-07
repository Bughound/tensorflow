import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 300, 300
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x) / 255
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)

  print(100 * np.max(tf.nn.softmax(array[0])))
  print(result)
  if answer == 0:
    print("pred: abejas")
  elif answer == 1:
    print("pred: arañas")
  elif answer == 2:
    print("pred: langostas")

  return answer

predict('imagenes/arañas/ia_100000002.jpeg')