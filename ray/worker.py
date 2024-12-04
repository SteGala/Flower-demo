import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import numpy as np
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from filelock import FileLock
import time

# Define the neural network model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to get data loaders
def get_data_loader():
    """Safely downloads data and returns training set dataloader."""
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = DataLoader(
            datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
            batch_size=128,
            shuffle=True,
        )
    return train_loader

def main():
    parser = argparse.ArgumentParser(description='Worker')
    parser.add_argument('--address', type=str, default=None, help='Ray cluster address (e.g., "auto" or "ray://<ip>:<port>")')
    parser.add_argument('--worker-id', type=str, default='worker_1', help='Unique ID for the worker')
    args = parser.parse_args()

    # Specify the same namespace
    namespace = 'ps_namespace'

    if args.address:
        ray.init(address=args.address, namespace=namespace)
    else:
        ray.init(namespace=namespace)

    # Get a handle to the ParameterServer
    ps = ray.get_actor("ParameterServer")

    train_loader = iter(get_data_loader())

    model = ConvNet()

    while True:
        weights = ray.get(ps.get_weights.remote())
        model.load_state_dict(weights)
        try:
            data, target = next(train_loader)
        except StopIteration:
            train_loader = iter(get_data_loader())
            data, target = next(train_loader)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        gradients = [p.grad.data.numpy() for p in model.parameters()]
        # Send gradients to ParameterServer
        ps.apply_gradients.remote(gradients)
        print(f"Worker {args.worker_id}: loss {loss.item()}")
        time.sleep(0.1)  # Sleep to prevent overloading the parameter server

if __name__ == '__main__':
    main()