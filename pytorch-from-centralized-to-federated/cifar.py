# cifar.py

"""
Enhanced PyTorch CIFAR-10 Image Classification with ResNet18.

This code loads the CIFAR-10 dataset and defines a ResNet18 model tailored for it,
suitable for federated learning using Flower.

Author: [Your Name]
Date: 2024-12-03
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Subset
import math


class ResNetClientModel(nn.Module):
    """ResNet18 model customized for CIFAR-10 classification."""

    def __init__(self) -> None:
        super(ResNetClientModel, self).__init__()
        # Load a larger ResNet model
        self.model = models.resnet50(pretrained=False, num_classes=10)
        # Modify the first convolution layer
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the max pooling layer
        self.model.maxpool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ResNet18."""
        return self.model(x)


def load_data(
    local_data_path: str = ".", partition_id: int = 0, num_partitions: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 data from a local directory and partition it for federated learning.

    Args:
        local_data_path (str): Path to the parent directory where CIFAR-10 is stored.
        partition_id (int): ID of the current partition (client).
        num_partitions (int): Total number of partitions (clients).

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing DataLoaders for the client.
    """
    local_data_path = "."
    # Define transformations
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load the full training and test datasets
    full_train_dataset = datasets.CIFAR10(
        root=local_data_path,
        train=True,
        download=False,  # Ensure dataset is downloaded
        transform=pytorch_transforms,
    )

    full_test_dataset = datasets.CIFAR10(
        root=local_data_path,
        train=False,
        download=False,
        transform=pytorch_transforms,
    )

    # Partition the training data
    train_size = len(full_train_dataset)
    partition_size = math.ceil(train_size / num_partitions)
    indices = list(range(train_size))
    start = partition_id * partition_size
    end = min(start + partition_size, train_size)
    subset_indices = indices[start:end]
    train_subset = Subset(full_train_dataset, subset_indices)

    # Optionally, partition the test data similarly
    # Here, we keep test data the same across clients
    test_subset = full_test_dataset  # Alternatively, partition test data as well

    # Create DataLoaders
    trainloader = DataLoader(
        train_subset, batch_size=128, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        test_subset, batch_size=128, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def train(
    net: nn.Module,
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
    net: nn.Module,
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
 