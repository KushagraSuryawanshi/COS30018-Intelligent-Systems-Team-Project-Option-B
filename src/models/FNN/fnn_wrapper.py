import os
from PIL import Image
import torch
import torchvision.transforms as transforms

"""
Wrapper for Feedforward Neural Network (FNN) inference.

Classes:
 - FNNModel: Provides preprocessing and inference functionality for FNN models.
"""

# Attempt to import the trained Net class from FNN.py
try:
    from src.models.FNN.FNN import Net  # Adjust import path as needed
except Exception:
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        """
        Backup Net class definition for inference if import fails.
        """

        def __init__(self):
            """Initialize the fallback fully connected FNN model."""
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 512)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 256)
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(256, 128)
            self.dropout3 = nn.Dropout(0.5)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            """
            Forward propagation through the fallback network.

            :param x: Input tensor of shape (batch_size, 784).
            :return: Log-probabilities tensor of shape (batch_size, 10).
            """
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = self.dropout3(x)
            x = self.fc4(x)
            return F.log_softmax(x, dim=1)


# Normalization constants used during FNN training
FNN_NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))


class FNNModel:
    """
    Wrapper class for loading, preprocessing, and predicting with a trained FNN model.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Load a trained FNN model and prepare for inference.

        :param model_path: Path to the trained FNN model (.pt file).
        :param device: Device string ("cpu" or "cuda"). Auto-detects if None.
        :raises FileNotFoundError: If the specified model file is missing.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = Net().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FNN model file not found: {model_path}")

        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Define transformation pipeline for inference
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x),  # Placeholder for optional preprocessing
            transforms.ToTensor(),
            FNN_NORMALIZE
        ])

    def _preprocess(self, crop_image):
        """
        Preprocess the input image for model inference.

        - Converts input to grayscale if needed.
        - Converts to tensor and normalizes.
        - Flattens to match FNN input shape (1, 784).

        :param crop_image: Input image as a NumPy array or PIL Image.
        :return: Preprocessed tensor on the correct device, shape (1, 784).
        """
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)

        # Convert to tensor (C, H, W)
        tensor = transforms.ToTensor()(crop_image)

        # Convert RGB to grayscale if needed
        if tensor.size(0) == 3:
            tensor = tensor.mean(dim=0, keepdim=True)

        tensor = FNN_NORMALIZE(tensor)
        tensor = tensor.view(1, -1).to(self.device)

        return tensor

    def predict(self, crop_image):
        """
        Predict the digit class from an input image.

        :param crop_image: PIL Image or NumPy array representing the digit image.
        :return: Tuple (label, confidence)
            - label (int): Predicted digit class (0–9).
            - confidence (float): Probability of the predicted class.
        """
        x = self._preprocess(crop_image)
        with torch.no_grad():
            out = self.model(x)  # Log-softmax output
            probs = torch.exp(out)
            conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())

    def predict_from_preprocessed(self, fnn_input_np):
        """
        Predict from a preprocessed and normalized input tensor.

        :param fnn_input_np: NumPy array of shape (1, 784), normalized with MNIST stats.
        :return: Tuple (label, confidence)
            - label (int): Predicted digit class (0–9).
            - confidence (float): Probability of the predicted class.
        """
        tensor = torch.from_numpy(fnn_input_np).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.exp(out)
            conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())