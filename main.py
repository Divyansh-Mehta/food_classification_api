from flask import Flask, request, jsonify
from PIL import Image
import io
from ultralytics import YOLO
import numpy as np


# categories = ["Bisibelebath", "Biryani",  "Butternaan", "chaat", "Idly", "Chappati", "Halwa", 
#               "Gulab Jamun", "Dhokla", "Dosa", "Vada Pav", "Upma", "Noodles", "Tandoori Chicken", 
#               "Poori", "Kathi Roll", "Samosa", "Meduvadai", "Paniyaram", "Ven Pongal"]


app = Flask(__name__)
model = YOLO("./best.pt")

def predict_image(img):
    results = model.predict(img)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    return names_dict[np.argmax(probs)]


@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})

        image = request.files['image']
        img = Image.open(io.BytesIO(image.read()))

        # Make a prediction
        prediction = predict_image(img)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str("hello" + e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
