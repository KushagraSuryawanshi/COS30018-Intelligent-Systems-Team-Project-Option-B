import os
import numpy as np
from PIL import Image
import tensorflow as tf

"""
Sequential model wrapper for prediction using trained TensorFlow models.

Classes:
 - SeqModel: Provides preprocessing and inference utilities for Sequential models.
"""


class SeqModel:
    """
    Wrapper class for TensorFlow Sequential models trained on MNIST-like datasets.

    Provides image preprocessing, normalization, and inference functions compatible
    with the training pipeline.
    """

    def __init__(self, model_path: str):
        """
        Initialize and load the trained Sequential model from the given path.

        :param model_path: Path to the saved Keras model (.keras or .h5 file).
        :raises FileNotFoundError: If the specified model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sequential model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.input_size = (28, 28)  # Input image resolution used during training

    def _preprocess(self, crop_image):
        """
        Preprocess input image for model prediction.

        - Converts to grayscale.
        - Resizes to 28x28 pixels.
        - Scales pixel values to [0, 1].
        - Ensures digits are white-on-black for MNIST consistency.
        - Applies normalization along axis 1.

        :param crop_image: PIL Image or NumPy array.
        :return: Normalized NumPy array of shape (1, 28, 28) as float32.
        """
        # Ensure image is a PIL Image
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)

        # Convert to grayscale and resize
        img = crop_image.convert("L").resize(self.input_size, Image.BILINEAR)

        # Scale pixel intensities to [0, 1]
        arr = np.asarray(img).astype("float32") / 255.0

        # Invert if background is white (ensure white digits on black)
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        # Apply same normalization as used in training
        arr = tf.keras.utils.normalize(arr, axis=1)

        # Model expects input shape (1, 28, 28)
        return arr.reshape(1, 28, 28).astype("float32")

    def predict(self, crop_image):
        """
        Predict the digit class for a given image.

        :param crop_image: Input image as PIL Image or NumPy array.
        :return: Tuple (label, confidence)
            - label (int): Predicted digit class (0–9).
            - confidence (float): Probability of the predicted class.
        """
        x = self._preprocess(crop_image)
        probs = self.model.predict(x)  # Output shape: (1, 10)
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])
        return pred, conf

    def predict_from_preprocessed(self, seq_input_np):
        """
        Predict from a preprocessed normalized NumPy array.

        :param seq_input_np: NumPy array of shape (1, 28, 28), float32 normalized input.
        :return: Tuple (label, confidence)
            - label (int): Predicted digit class (0–9).
            - confidence (float): Probability of the predicted class.
        """
        probs = self.model.predict(seq_input_np)  # Output shape: (1, 10)
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])
        return pred, conf