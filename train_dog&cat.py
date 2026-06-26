import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# IMAGE SIZE
img_size = 150

# DATA AUGMENTATION + SPLIT
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# TRAIN DATA
train = datagen.flow_from_directory(
    "train3",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# VALIDATION DATA (FIXED: same folder as train3)
val = datagen.flow_from_directory(
    "test3",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# MODEL ARCHITECTURE
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

# COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# TRAIN MODEL
model.fit(
    train,
    validation_data=val,
    epochs=10
)

# SAVE MODEL
model.save("my_model.keras", include_optimizer=False)

print("✅ Model trained and saved successfully!")