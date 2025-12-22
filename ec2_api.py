from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uuid
import os

app = Flask(__name__)
CORS(app)

model = load_model('FV.h5')

labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper',
    9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
    14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
    19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear',
    24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato',
    28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruit_list = [
    'apple','banana','bell pepper','chilli pepper','grapes','jalepeno',
    'kiwi','lemon','mango','orange','paprika','pear','pineapple',
    'pomegranate','watermelon'
]

calorie_data = {
    'apple':'52',
    'banana':'89',
    'carrot':'41',
    'potato':'87',
    'tomato':'18',
    'mango':'60',
    'orange':'47',
    'pineapple':'50',
    'pear':'57',
    'peas':'81',
    'cabbage':'25',
    'corn':'86',
}

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    image_path = f"temp_{uuid.uuid4()}.jpg"
    file.save(image_path)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, 0)

    pred = model.predict(img)
    idx = np.argmax(pred)

    name = labels[idx]
    os.remove(image_path)

    category = "Fruit" if name in fruit_list else "Vegetable"

    calories = calorie_data.get(name, "Unknown")

    return jsonify({
        "prediction": name.capitalize(),
        "category": category,
        "calories": calories + " kcal / 100g",
        "advice": f"{name.capitalize()} is healthy and nutritious."
    })

if __name__ == "__main__":
    app.run(debug=True)
