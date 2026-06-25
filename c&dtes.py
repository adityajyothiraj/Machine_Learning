import cv2 as cv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

loaded_model = tf.keras.models.load_model('my_model.keras')

img_size = 150

def predict_image(image_path):
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = loaded_model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"{image_path}:Dog(Confidence: {prediction:.2f})")
    else:
        print(f"{image_path}:Cat(Confidence: {1 - prediction:.2f})")

#image = cv.imread('pexels-photo-32557420.jpeg')
predict_image('tes/maltese-1123016_640.jpg')