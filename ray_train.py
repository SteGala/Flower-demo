import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ray
import numpy as np
from filelock import FileLock

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
    """Safely downloads data and returns training/validation set dataloaders."""
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=128,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128,
            shuffle=True,
        )
    return train_loader, test_loader

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
    def __init__(self, lr=1e-2):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        for g, p in zip(summed_gradients, self.model.parameters()):
            p.grad = torch.from_numpy(g)
        self.optimizer.step()
        return self.model.state_dict()

    def get_weights(self):
        return self.model.state_dict()

# Define the Data Worker
@ray.remote
class DataWorker:
    def __init__(self):
        self.model = ConvNet()
        self.data_loader = iter(get_data_loader()[0])

    def compute_gradients(self, weights):
        self.model.load_state_dict(weights)
        try:
            data, target = next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(get_data_loader()[0])
            data, target = next(self.data_loader)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        gradients = [p.grad.data.numpy() for p in self.model.parameters()]
        return gradients

# Initialize Ray
ray.init()

# Hyperparameters
iterations = 200
num_workers = 2

# Create a parameter server
ps = ParameterServer.remote()

# Create workers
workers = [DataWorker.remote() for _ in range(num_workers)]

# Training loop
print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
for i in range(iterations):
    gradients = [
        worker.compute_gradients.remote(current_weights) for worker in workers
    ]
    current_weights = ps.apply_gradients.remote(*gradients)
    if i % 10 == 0:
        model = ConvNet()
        model.load_state_dict(ray.get(current_weights))
        accuracy = evaluate(model, get_data_loader()[1])
        print(f"Iter {i}: \taccuracy is {accuracy:.1f}")

print("Training complete.")
