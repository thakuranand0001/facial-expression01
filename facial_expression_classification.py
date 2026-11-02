# Facial Expression Classification using TensorFlow CNN
# Dataset folders:
# facial-expression/
#   train/{angry,disgust,fear,happy,neutral,sad,surprise}/
#   test/{angry,disgust,fear,happy,neutral,sad,surprise}/

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

print("TensorFlow version:", tf.__version__)

# 1. Dataset paths and settings
DATA_DIR = Path("facial-expression")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 30

# 2. Load images from folders
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_data.class_names
print("Expression classes:", class_names)
assert len(class_names) == 7, "Please provide all seven expression folders."

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Display sample images
plt.figure(figsize=(10, 8))
for images, labels in train_data.take(1):
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# 4. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 6. Plot accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 7. Evaluate model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 8. Predict expression for one image
def predict_expression(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    print("Predicted expression:", class_names[predicted_index])
    print(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

    plt.imshow(image)
    plt.title(
        f"{class_names[predicted_index]} "
        f"({predictions[predicted_index] * 100:.1f}%)"
    )
    plt.axis("off")
    plt.show()

# Example: update this path to a real image in your test folder
# predict_expression("facial-expression/test/happy/example.jpg")

# 9. Save model and class labels
model.save("facial_expression_model.keras")

with open("classes.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file)

print("Model saved as facial_expression_model.keras")
print("Class labels saved as classes.json")
