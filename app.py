import os
import requests
import numpy as np
import tensorflow as tf

from imageio import imwrite, imread_v2
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify, render_template

print(tf.__version__)


with open("fashion_model.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model.h5")
model.summary()


app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/<string:img_name>", methods=["POST", "GET"])
def classify_image(img_name):
    upload_dir = "uploads/"
    image = imread_v2(upload_dir + img_name)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    
    prediction = model.predict([image.reshape(1, 28*28)])

    return jsonify({"object_indentified": classes[np.argmax(prediction[0])]})

if __name__ == '__main__':
    app.run(debug=False)
