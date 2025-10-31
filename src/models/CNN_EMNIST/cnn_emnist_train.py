import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.CNN_EMNIST.model import Net


"""
CNN+ Training Script for EMNIST ByClass classification (62 classes)
"""

# === Canonical directories ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "saved_models")
DATA_DIR = os.path.expanduser("~/.pytorch/emnist")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Training parameters
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EMNIST] Using device: {DEVICE}")

# Normalization constants used for EMNIST
EMNIST_MEAN = 0.5
EMNIST_STD = 0.5


def get_dataloaders(batch_size=DEFAULT_BATCH_SIZE):
    """Prepare EMNIST ByClass dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,))
    ])

    train_set = datasets.EMNIST(
        root=DATA_DIR, split="byclass", train=True, download=True, transform=transform
    )
    test_set = datasets.EMNIST(
        root=DATA_DIR, split="byclass", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def evaluate(model, loader, criterion):
    """Evaluate model performance."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_main(epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE, out_dir=None):
    """Train the EMNIST CNN+ model."""
    out_dir = out_dir or MODELS_DIR
    os.makedirs(out_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size)
    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"[EMNIST] Epoch {epoch}/{epochs} "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={test_loss:.4f}, Acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(out_dir, "cnn_emnist_byclass.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"[EMNIST] Saved new best model -> {best_model_path}")

    print(f"[EMNIST] Training complete. Best accuracy={best_acc:.4f}")
    return os.path.join(out_dir, "cnn_emnist_byclass.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    train_main(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dir=args.out_dir
    )