import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt

# LOAD TRAINED MODEL
model = load_model("inception_v3_model.keras")
# OR
# model = load_model("face_emotion_final.keras")

print("Model loaded successfully!")

# LOAD CLASS NAMES FROM TRAINING DATASET
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = datagen.flow_from_directory(
    r"Face_Emotion_Dataset\train",
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())

print("\nClass Names:")
print(class_names)

# IMAGE TO TEST
img_path = r"Face_Emotion_Dataset\test\surprise\2804.jpg"

# Change the above path to your image

# LOAD IMAGE
img = image.load_img(
    img_path,
    target_size=(299, 299)
)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# PREPROCESS IMAGE
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Same preprocessing used during training
img_array = preprocess_input(img_array)

# MAKE PREDICTION
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = np.max(prediction) * 100

# DISPLAY RESULT
print("\nPrediction")
print("--------------------------------")
print("Predicted Emotion :", predicted_class)
print("Confidence        : {:.2f}%".format(confidence))

# DISPLAY ALL PROBABILITIES
print("\nClass Probabilities")
print("--------------------------------")

for i, emotion in enumerate(class_names):
    print(f"{emotion:15s}: {prediction[0][i] * 100:.2f}%")

# BAR CHART OF PROBABILITIES
plt.figure(figsize=(8,5))
bars = plt.bar(class_names, prediction[0] * 100)
plt.title("Emotion Prediction Probabilities")
plt.xlabel("Emotion")
plt.ylabel("Probability (%)")
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.1f}%",
        ha='center'
    )

plt.tight_layout()
plt.show()