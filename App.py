import streamlit as st
from PIL import Image
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Ensure upload directory exists
if not os.path.exists('./upload_images'):
    os.makedirs('./upload_images')

# Load model
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

fruits = [f.capitalize() for f in [
    'apple', 'banana', 'bell pepper', 'chilli pepper', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'mango', 'orange', 'paprika', 'pear',
    'pineapple', 'pomegranate', 'watermelon'
]]

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        return scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    except:
        return None

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    return labels[np.argmax(answer)].capitalize()

def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file:
        img = Image.open(img_file).resize((250, 250))
        st.image(img)

        save_path = os.path.join('./upload_images', img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = prepare_image(save_path)
        category = "Fruit" if result in fruits else "Vegetable"

        st.info(f'**Category : {category}**')
        st.success(f'**Predicted : {result}**')

        cal = fetch_calories(result)
        if cal:
            st.warning(f'**{cal} (100 grams)**')

if __name__ == '__main__':
    run()
