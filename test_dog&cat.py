import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# LOAD TRAINED MODEL
model = load_model("my_model.keras")

# CLASS LABELS (IMPORTANT: match folder order from training)
# You can check order using: train.class_indices during training
class_labels = ["CAT", "DOG"]  # change to your real folder names

img_size = 150

def predict_image(img_path):
    # CHECK FILE EXISTS
    if not os.path.exists(img_path):
        print("File not found:", img_path)
        return

    # LOAD IMAGE
    img = image.load_img(img_path, target_size=(img_size, img_size))

    # CONVERT TO ARRAY
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # NORMALIZE
    img_array = img_array / 255.0

    # PREDICT
    prediction = model.predict(img_array)

    # BINARY OUTPUT (sigmoid)
    score = prediction[0][0]

    if score > 0.5:
        predicted_class = class_labels[1]
    else:
        predicted_class = class_labels[0]

    print("Prediction:", predicted_class)
    print("Confidence:", float(score) * 100, "%")

    return predicted_class, score

img_path = "train3\cats\cat_1.jpg"   # CHANGE THIS TO YOUR IMAGE

predict_image(img_path)