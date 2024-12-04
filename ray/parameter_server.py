import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import numpy as np
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    """Safely downloads data and returns validation set dataloader."""
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, download=True, transform=mnist_transforms),
        batch_size=128,
        shuffle=True,
    )
    return test_loader

# Function to evaluate the model
def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

# Define the Parameter Server
@ray.remote
class ParameterServer:
    def __init__(self, lr=1e-2, num_workers=2):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.num_workers = num_workers
        self.gradients = []
        self.iteration = 0

    def apply_gradients(self, gradients):
        self.gradients.append(gradients)
        if len(self.gradients) >= self.num_workers:
            # Sum gradients from all workers
            summed_gradients = [
                np.stack(gradient_zip).sum(axis=0)
                for gradient_zip in zip(*self.gradients)
            ]
            self.optimizer.zero_grad()
            for g, p in zip(summed_gradients, self.model.parameters()):
                p.grad = torch.from_numpy(g)
            self.optimizer.step()
            self.gradients = []
            self.iteration += 1
            print(f"Updated model at iteration {self.iteration}")

    def get_weights(self):
        return self.model.state_dict()

    def get_iteration(self):
        return self.iteration


def main():
    parser = argparse.ArgumentParser(description='Parameter Server')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
    parser.add_argument('--address', type=str, default=None, help='Ray cluster address (e.g., "auto" or "ray://<ip>:<port>")')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    args = parser.parse_args()

    # Specify a namespace
    namespace = 'ps_namespace'

    if args.address:
        ray.init(address=args.address, namespace=namespace)
    else:
        ray.init(namespace=namespace)

    # Create a named Parameter Server actor
    ps = ParameterServer.options(name="ParameterServer").remote(lr=args.lr, num_workers=args.num_workers)

    test_loader = get_data_loader()

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}")
        while True:
            # Wait for all workers to contribute gradients
            current_iteration = ray.get(ps.get_iteration.remote())
            if current_iteration >= (epoch + 1) * 100:  # Assuming 100 iterations per epoch
                break
            import time
            time.sleep(1)

        # Evaluate the model
        current_weights = ray.get(ps.get_weights.remote())
        model = ConvNet()
        model.load_state_dict(current_weights)
        accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: accuracy is {accuracy:.1f}%")

    print("Training complete.")

if __name__ == '__main__':
    main()