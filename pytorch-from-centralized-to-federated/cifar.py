"""
Simplified PyTorch CIFAR-10 Image Classification with an Even Simpler CNN.

This code is adapted to load the CIFAR-10 dataset from a local directory
and uses an even more simplified convolutional neural network suitable for CPU training.

Author: [Your Name]
Date: 2024-12-02
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


class SimpleNet(nn.Module):
    """Even More Simplified CNN for CIFAR-10 classification."""

    def __init__(self) -> None:
        super(SimpleNet, self).__init__()
        # Single convolutional layer with fewer filters
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)  # Output: 4 x 32 x 32
        self.pool = nn.MaxPool2d(2, 2)               # Output: 4 x 16 x 16
        # Global Average Pooling instead of fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: 4 x 1 x 1
        self.fc = nn.Linear(4, 10)                        # Output layer

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = F.relu(self.conv1(x))  # Apply convolution and activation
        x = self.pool(x)           # Apply pooling
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = x.view(-1, 4)           # Flatten
        x = self.fc(x)              # Output layer
        return x


def load_data(local_data_path: str = "./data") -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 data from a local directory.

    Args:
        local_data_path (str): Path to the parent directory where CIFAR-10 is stored.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    # Define transformations
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load training and test datasets from local directory
    train_dataset = datasets.CIFAR10(
        root=local_data_path,
        train=True,
        download=False,  # Set to False since dataset is manually downloaded
        transform=pytorch_transforms,
    )

    test_dataset = datasets.CIFAR10(
        root=local_data_path,
        train=False,
        download=False,
        transform=pytorch_transforms,
    )

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    return trainloader, testloader


def train(
    net: SimpleNet,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the network on the training data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            if batch_idx % 100 == 99:  # Print every 100 mini-batches
                print(
                    f"[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

    print("Finished Training")


def test(
    net: SimpleNet,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return average_loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print("Centralized PyTorch training with an even more simplified CNN")
    print("Loading data...")
    # Set local_data_path to the parent directory containing 'cifar-10-batches-py'
    local_data_path = "/home/cc/Flower-demo"
    trainloader, testloader = load_data(local_data_path=local_data_path)
    net = SimpleNet()
    print("Starting training...")
    train(net=net, trainloader=trainloader, epochs=10, device=DEVICE)
    print("Evaluating model...")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print(f"Test Loss: {loss:.3f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()