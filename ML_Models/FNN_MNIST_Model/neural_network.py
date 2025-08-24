import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# Project Configuration
project_path = '.'
print(f"Project path is set to: {os.path.abspath(project_path)}")


# Helper Functions
def load_image(image_path):
    """Loads, normalizes, and flattens images loaded """
    image = Image.open(image_path)
    normalize_image = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081,))
    ])
    image_normalized = normalize_image(image)
    image_normalized = image_normalized.view(-1, 28 * 28)
    return image_normalized


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512) # 512 neurons in the first layer
        self.dropout1 = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(512, 256) # 256 neurons in the second layer
        self.dropout2 = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc3 = nn.Linear(256, 128) # 128 neurons in the third layer
        self.dropout3 = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc4 = nn.Linear(128, 10) # 10 neurons in the fourth layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def create_nn(learning_rate=0.05, batch_size=200, epochs=10,
              log_interval=10, saving_name="net.pt"):
    print("Loading training set")
    # Data augmentation applied to the training set
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Randomly rotate, translate, and scale the image slighly
        transforms.GaussianBlur(kernel_size=3), # Slight gaussian blur to reduce noise with hand drawn images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=train_transform),
        batch_size=batch_size, shuffle=True)

    print("Loading data test set")
    # No augmentation on the test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081,))
    ])

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=test_transform),
        batch_size=batch_size, shuffle=True)

    print("Creating Neural Network")
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # Add the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.NLLLoss()

    print("Beginning Training")
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
        # Step the scheduler after each epoch
        scheduler.step()

    print("Starting Testing")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    torch.save(net.state_dict(), os.path.join(project_path, saving_name))


def guess_number(image_path, loading_name):
    model_path = os.path.join(project_path, loading_name)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. "
              "Please create the neural network first (by typing 'nn').")
        return None

    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    image_tensor = load_image(image_path)
    output = net(image_tensor)
    pred = output.data.max(1)[1].item()
    return str(pred)


def start(prompt):
    """Main function to start the application with a user input prompt """
    print("Starting Processes")
    print("Create Neural Network (nn) OR Guess for a Directory (gd)?")
    task = input(prompt)
    if task == "nn":
        print("Creating Neural Network for MNIST")
        create_nn()
    elif task == "gd":
        test_dir = os.path.join(project_path, "test_images")

        if not os.path.exists(test_dir):
            print(f"Error: Directory not found at {test_dir}.")
            print("Please create a folder named 'test_images' in your "
                  "project directory and upload your files.")
            return

        print("Guessing digits from multiple image files...")

        image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]

        if not image_files:
            print(f"No .png files found in {test_dir}.")
            return

        for image_file in sorted(image_files):
            image_path = os.path.join(test_dir, image_file)
            predicted_character = guess_number(image_path, "net.pt")
            if predicted_character is not None:
                print(f"File: {image_file} -> Predicted digit: {predicted_character}")
    else:
        print("Invalid input. Please enter 'nn' or 'gd'.")


if __name__ == "__main__":
    start("> ")