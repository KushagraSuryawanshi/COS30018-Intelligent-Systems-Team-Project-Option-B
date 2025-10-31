import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Train and save a simple Sequential neural network using the MNIST dataset.

This script:
  - Loads and normalizes MNIST data
  - Defines a small dense (fully connected) model
  - Trains the model for several epochs
  - Saves the trained model in Keras format
  - Optionally evaluates the trained model on the training set
"""

# Define canonical model save path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../..", "models", "saved_models")
)
os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_PATH = os.path.join(MODELS_DIR, "Sequential.keras")
print(f"[SEQ] Save directory: {SAVE_PATH}")

# Load MNIST dataset (handwritten digits 0–9)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define the Sequential model
model = keras.Sequential([
    layers.Input(shape=(28, 28), name="inputs"),  # Input: 28x28 grayscale image
    layers.Flatten(),                             # Flatten to 1D vector
    layers.Dense(128, activation="relu"),         # Hidden layer 1
    layers.Dense(128, activation="relu"),         # Hidden layer 2
    layers.Dense(10, activation="softmax"),       # Output layer (10 digits)
])

# Compile model with optimizer, loss function, and metrics
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x_train.astype("float32"),
    y_train.astype("int32"),
    epochs=3
)

# Save the trained model
model.save(SAVE_PATH)
print(f"[SEQ] Model saved successfully at: {SAVE_PATH}")

# Optional: quick evaluation on training data
model = tf.keras.models.load_model(SAVE_PATH)
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print(f"[SEQ] Evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")