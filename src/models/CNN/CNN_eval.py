import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils

"""
CNN Evaluation Script

Provides utilities for evaluating a trained CNN model on MNIST
or cropped digit images, visualizing predictions, and saving results.

Classes:
 - SimpleCNN: Basic CNN architecture for digit classification.

Functions:
 - load_image_as_tensor(path)
 - build_crops_batch(crops_dir, limit=25)
 - save_crops_grid_with_preds(images, preds, files, save_path)
 - main()
"""


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for digit classification.
    """

    def __init__(self):
        """Initialize convolutional, pooling, and fully connected layers."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        Forward pass through the CNN.

        :param x: Input tensor of shape (N, 1, 28, 28).
        :return: Output tensor of shape (N, 10).
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


MNIST_NORM = transforms.Normalize((0.1307,), (0.3081,))


def load_image_as_tensor(path: str):
    """
    Load and normalize an image for CNN inference.

    - Converts to grayscale.
    - Resizes to 28x28 pixels.
    - Inverts if background is bright.
    - Normalizes using MNIST mean and standard deviation.

    :param path: Path to image file.
    :return: Normalized torch tensor of shape (1, 28, 28).
    """
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    x = np.array(img).astype(np.float32) / 255.0

    # Invert if background is white (digits should be white on black)
    if x.mean() > 0.5:
        x = 1.0 - x

    # Convert to tensor and normalize
    x = torch.from_numpy(x).unsqueeze(0)
    x = MNIST_NORM(x)
    return x


def build_crops_batch(crops_dir: str, limit: int = 25):
    """
    Build a batch tensor from cropped digit images.

    Searches the provided directory for image files, loads up to `limit`,
    and returns a tensor suitable for CNN inference.

    :param crops_dir: Directory containing cropped digit images.
    :param limit: Maximum number of images to include in the batch.
    :return: Tuple (batch_tensor, file_list)
        - batch_tensor (torch.Tensor): Tensor of shape (N, 1, 28, 28).
        - file_list (list): Corresponding file paths.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(crops_dir, ext)))

    # Prefer segmented_* images if available
    seg_files = sorted([
        f for f in files if os.path.basename(f).startswith("segmented_")
    ])
    if seg_files:
        files = seg_files
    files = files[:limit]

    tensors = [load_image_as_tensor(p) for p in files]
    if not tensors:
        return None, []

    batch = torch.stack(tensors, dim=0)
    return batch, files


def save_crops_grid_with_preds(images, preds, files, save_path):
    """
    Save a grid of images annotated with their predicted labels.

    :param images: Batch tensor of images (N, 1, 28, 28).
    :param preds: Tensor of predicted labels (N,).
    :param files: List of corresponding file paths.
    :param save_path: Path to save the output image grid.
    :return: None
    """
    # De-normalize for display
    imgs_disp = images.clone()
    imgs_disp = imgs_disp * 0.3081 + 0.1307

    # Arrange in a square grid
    grid = utils.make_grid(
        imgs_disp,
        nrow=int(np.ceil(np.sqrt(images.size(0)))),
        padding=2
    )
    npimg = grid.numpy().transpose(1, 2, 0).squeeze()

    plt.figure(figsize=(6, 6))
    plt.imshow(npimg, cmap="gray")
    plt.axis("off")
    title = "Cropped digits predictions: " + " ".join(
        str(p) for p in preds.tolist()
    )
    plt.title(title)

    # Print filename → prediction mapping for reference
    for f, p in zip(files, preds.tolist()):
        print(f"[Pred] {os.path.basename(f)} -> {p}")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """
    Entry point for evaluating CNN on cropped digits or MNIST test set.

    - Loads model weights.
    - Optionally evaluates accuracy on MNIST test set.
    - Performs inference on segmented digit images.
    - Saves a prediction grid image to the output directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("../..", "models", "saved_models", "cnn_model_best.pth"),
        help="Path to trained model weights."
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default="../outputs",
        help="Directory containing cropped digit images."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../outputs",
        help="Directory to save output predictions."
    )
    parser.add_argument(
        "--eval_mnist",
        action="store_true",
        help="If set, evaluate accuracy on MNIST test dataset."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load model
    model = SimpleCNN().to(device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[Info] Loaded model weights from {args.model}")

    # Optionally evaluate on MNIST test set
    if args.eval_mnist:
        from torch.utils.data import DataLoader
        test_set = datasets.MNIST(
            root="~/.pytorch/mnist",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                MNIST_NORM
            ])
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[MNIST Test] accuracy = {correct / total: .4f}")

    # Inference on cropped digits
    batch, files = build_crops_batch(args.crops_dir, limit=25)
    if batch is None:
        print(f"[Warn] No images found in {args.crops_dir}. "
              f"Expected files like 'segmented_00.png' from Task 2.")
        return

    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu()

    save_path = os.path.join(args.out_dir, "sample_preds.png")
    save_crops_grid_with_preds(batch.cpu(), preds, files, save_path)
    print(f"[Info] Saved {save_path}")


if __name__ == "__main__":
    main()