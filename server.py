#
# TensorFlow Image classification prediction server
#
# Author: José Luis Pereira
#

import time
from absl import app, logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# Server configuration
predict_labels = ['abeja', 'araña', 'langosta'] # Hardcode labels prediction. TODO: Refactor this to load a label file, need to rewrite train.py code to store the labels.
model_path = './modelo/modelo.h5'
weights_path = './modelo/pesos.h5'
size = 300
port = 5000

# Allow to expand the memory used for the graphic card.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

cnn = load_model(model_path)
cnn.load_weights(weights_path)


# Initialize Flask application
app = Flask(__name__)

@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        print(image)
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = load_img(image_name, target_size=(size,size))
        raw_images.append(img_raw)
        
    num = 0
    
    response = []

    for j in range(len(raw_images)):
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = img_to_array(raw_img) / 255
        img = np.expand_dims(img, axis=0)

        t1 = time.time()
        array = cnn.predict(img)
        result = array[0]
        answer = np.argmax(result)

        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for index, val in enumerate(result):
            print(index)
            print(val)
            responses.append({
                "class": predict_labels[index],
                "confidence": float("{0:.2f}".format(np.array(result[index])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })

    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=port)