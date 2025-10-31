import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from src.models.CNN_EMNIST.model import Net

# EMNIST ByClass label map (62 classes: 0-9, A-Z, a-z)
EMNIST_BYCLASS_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

class EMNISTModel:
    """Wrapper for CNN+ EMNIST ByClass model (62 classes)."""

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize EMNIST model wrapper.

        Args:
            model_path: Path to saved .pth file
            device: torch device ('cpu' or 'cuda', auto-detected if None)

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Net().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"EMNIST model weights not found: {model_path}")

        # Load state dict
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            self.model.load_state_dict(state)

        self.model.eval()

        # EMNIST uses (0.5, 0.5) normalization as per training
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _preprocess(self, crop_image):
        """
        Preprocess image to model input format.

        Args:
            crop_image: PIL Image or numpy array (H,W) or (H,W,3)

        Returns:
            torch.Tensor: shape (1,1,28,28) on device
        """
        if not isinstance(crop_image, Image.Image):
            crop_image = Image.fromarray(crop_image)

        img = crop_image.convert("L").resize((28, 28), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0

        # Ensure white-on-black polarity (digits white on black background)
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        # Convert back to PIL for transform
        img2 = Image.fromarray((arr * 255).astype("uint8"))
        t = self.transform(img2)  # (1, 28, 28)

        return t.unsqueeze(0).to(self.device)  # (1, 1, 28, 28)

    def predict(self, crop_image):
        """
        Predict character from crop image.

        Args:
            crop_image: PIL Image or numpy array

        Returns:
            Tuple[str, float]: (predicted_character, confidence)
        """
        x = self._preprocess(crop_image)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            pred_idx = int(pred_idx.item())
            label = EMNIST_BYCLASS_LABELS[pred_idx] if pred_idx < len(EMNIST_BYCLASS_LABELS) else "?"

            return label, float(conf.item())

    def predict_from_preprocessed(self, emnist_tensor):
        """
        Predict from already-preprocessed tensor.
        Used for ensemble predictions to avoid redundant preprocessing.

        Args:
            emnist_tensor: torch.Tensor shape (1,1,28,28) on device

        Returns:
            Tuple[str, float]: (predicted_character, confidence)
        """
        with torch.no_grad():
            x = emnist_tensor.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            pred_idx = int(pred_idx.item())
            label = EMNIST_BYCLASS_LABELS[pred_idx] if pred_idx < len(EMNIST_BYCLASS_LABELS) else "?"

            return label, float(conf.item())