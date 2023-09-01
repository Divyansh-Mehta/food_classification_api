from flask import Flask, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image


categories = ["Bisibelebath", "Biryani",  "Butternaan", "chaat", "Idly", "Chappati", "Halwa", 
              "Gulab Jamun", "Dhokla", "Dosa", "Vada Pav", "Upma", "Noodles", "Tandoori Chicken", 
              "Poori", "Kathi Roll", "Samosa", "Meduvadai", "Paniyaram", "Ven Pongal"]

img_size = 256
model = load_model('./cnn_model.hdf5')

def preprocess_image(file):
    img = Image.open(file.stream)
    img = np.array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img


app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict_image_class():
    file = request.files['image']
    img = preprocess_image(file)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return jsonify({'msg': 'success', 'size': categories[predicted_class]})

if __name__ == "__main__":
    app.run(host="0.0.0.0")

