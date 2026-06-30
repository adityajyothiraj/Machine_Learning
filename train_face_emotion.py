# pip install tensorflow==2.13.0
# pip install pillow==10.3.0

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow Version:", tf.__version__)


# IMAGE SIZE
IMG_ROWS = 299
IMG_COLS = 299
BATCH_SIZE = 32
EPOCHS = 8

# DATASET PATHS
train_data_dir = r"Face_Emotion_Dataset\train"
validation_data_dir = r"Face_Emotion_Dataset\test"

# LOAD PRETRAINED MODEL
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_ROWS, IMG_COLS, 3)
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# CUSTOM CLASSIFIER
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)

predictions = Dense(5, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# DATA GENERATORS
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass Indices")
print(train_generator.class_indices)

# CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")
print(class_weights)

# CALLBACKS
checkpoint = ModelCheckpoint(
    "inception_v3_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.2,
    patience=5,
    min_lr=1e-5,
    verbose=1,
    mode="max"
)

callbacks = [checkpoint, earlystop, reduce_lr]

# COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# TRAIN MODEL
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# SAVE FINAL MODEL
model.save("face_emotion_final.keras")

print("\nModel saved successfully.")

# EVALUATE MODEL
loss, accuracy = model.evaluate(validation_generator)

print("\nValidation Loss :", loss)
print("Validation Accuracy :", accuracy)

# TRAINING GRAPH
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()

# CONFUSION MATRIX
validation_generator.reset()

predictions = model.predict(validation_generator)

y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(validation_generator.classes, y_pred)

plt.figure(figsize=(8,8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(validation_generator.class_indices.keys())
)

disp.plot(cmap=plt.cm.Blues, values_format="d")

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nTraining Completed Successfully!")