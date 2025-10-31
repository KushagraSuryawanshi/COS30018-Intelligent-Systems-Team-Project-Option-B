import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Sequential MNIST model training script using TensorFlow and Keras.

Functions:
 - preprocess_fn(image, label=None)
 - build_mnist_dataset(batch_size=128, extra_dir=None)
 - build_model()
 - train_and_save(epochs=3, batch_size=128, extra_dir=None)
"""

# Define base directories for model saving
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)
SAVE_DIR = os.path.join(MODELS_DIR, "Sequential.keras")

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_fn(image, label=None):
    """
    Preprocess an input image for MNIST-like sequential training.

    - Converts image to grayscale if necessary.
    - Resizes to 28x28 pixels.
    - Normalizes to range [0, 1].
    - Inverts if background is brighter than foreground.
    - Applies per-sample normalization along axis 1.

    :param image: Input image tensor (HxW or HxWx3).
    :param label: Optional label tensor.
    :return: Tuple (image, label) if label provided, else image tensor.
    """
    # Convert to grayscale if 3-channel image
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)

    # Resize and scale to [0, 1]
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Invert if mean brightness indicates white background
    mean = tf.reduce_mean(image)
    image = tf.cond(mean > 0.5, lambda: 1.0 - image, lambda: image)

    # Apply normalization per sample along axis 1
    image = tf.keras.utils.normalize(image, axis=1)

    if label is None:
        return image

    return image, label


def _create_tf_dataset(x_data, y_data, batch_size, shuffle=False):
    """
    Helper function to build a TensorFlow dataset with preprocessing.

    :param x_data: Input image data as NumPy array.
    :param y_data: Corresponding label array.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle the dataset.
    :return: Preprocessed TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_mnist_dataset(batch_size=128, extra_dir=None):
    """
    Build TensorFlow datasets for MNIST training and evaluation.

    - Loads the MNIST dataset.
    - Applies preprocessing and batching.
    - Optionally includes extra labeled data from a directory with subfolders 0–9.

    :param batch_size: Batch size for training and testing datasets.
    :param extra_dir: Optional path to an additional dataset directory.
    :return: Tuple (train_ds, test_ds) as preprocessed TensorFlow datasets.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Add channel dimension for grayscale format
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # Prepare training and test datasets
    train_ds = _create_tf_dataset(x_train, y_train, batch_size, shuffle=True)
    test_ds = _create_tf_dataset(x_test, y_test, batch_size)

    # Optionally add extra labeled data (expects subfolders 0–9)
    if extra_dir:
        extra_dir = str(Path(extra_dir).resolve())
        if os.path.isdir(extra_dir):
            extra_ds = tf.keras.preprocessing.image_dataset_from_directory(
                extra_dir,
                labels="inferred",
                label_mode="int",
                color_mode="grayscale",
                batch_size=batch_size,
                image_size=(28, 28),
                shuffle=True
            ).map(
                lambda x, y: (
                    tf.keras.utils.normalize(
                        tf.cast(x / 255.0, tf.float32), axis=1
                    ),
                    y,
                )
            )

            # Concatenate datasets
            train_ds = train_ds.concatenate(extra_ds)
            print(f"[SEQ] Added extra data from {extra_dir}")
        else:
            print(f"[SEQ] Extra dir {extra_dir} not found; skipping extra data.")

    return train_ds, test_ds


def build_model():
    """
    Build a simple sequential neural network for MNIST classification.

    Architecture:
     - Input: 28x28 grayscale image.
     - Two hidden layers (Dense 128, ReLU activation).
     - Output: Dense(10, softmax).

    :return: Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28), name="inputs"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_save(epochs=3, batch_size=128, extra_dir=None):
    """
    Train the MNIST model and save it to disk.

    :param epochs: Number of epochs to train the model.
    :param batch_size: Size of each training batch.
    :param extra_dir: Optional directory containing extra labeled data.
    :return: Path to the saved model file.
    """
    train_ds, test_ds = build_mnist_dataset(batch_size=batch_size, extra_dir=extra_dir)
    model = build_model()
    model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    # Save model
    model.save(SAVE_DIR)
    print(f"[SEQ] Model saved to {SAVE_DIR}")

    # Evaluate final model
    loss, acc = model.evaluate(test_ds)
    print(f"[SEQ] Final test eval -> loss: {loss:.4f}, acc: {acc:.4f}")

    return SAVE_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--extra-data",
        type=str,
        default=None,
        help="Optional: path to extra labeled data with subfolders 0–9",
    )

    args = parser.parse_args()

    train_and_save(
        epochs=args.epochs,
        batch_size=args.batch_size,
        extra_dir=args.extra_data,
    )