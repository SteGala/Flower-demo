# client.py

"""Flower client example using PyTorch with ResNet18 for CIFAR-10 image classification."""

import argparse
from collections import OrderedDict
import os
from typing import Dict, List, Tuple

import cifar  # Updated cifar.py with ResNetClientModel
import flwr as fl
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

disable_progress_bar()

USE_FEDBN: bool = False  # Set to True if using BatchNorm layers in the model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using ResNet18."""

    def __init__(
        self,
        model: cifar.ResNetClientModel,
        trainloader: DataLoader,
        testloader: DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy().astype(np.float32)
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy().astype(np.float32) for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)  # Minimal training
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, initialize ResNet18 model, and start CifarClient."""
    server_ip, server_port, partition_id, num_partitions = read_env_variable()

    # Load data with partitioning
    local_data_path = "./data"
    trainloader, testloader = cifar.load_data(local_data_path, partition_id, num_partitions)

    # Load ResNet18 model
    model = cifar.ResNetClientModel().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm (if used)
    try:
        inputs, _ = next(iter(trainloader))
        _ = model(inputs.to(DEVICE))
    except StopIteration:
        print("Train loader is empty. Please check data partitioning.")
        return

    # Start client
    client = CifarClient(model, trainloader, testloader)
    fl.client.start_client(server_address=f"{server_ip}:{server_port}", client=client)


def read_env_variable():
    """
    Read environment variables for server configuration and client partitioning.

    Returns:
        Tuple[str, str, int, int]: server_ip, server_port, partition_id, num_partitions
    """
    # Get node ID and total number of partitions (clients)
    server_ip = os.getenv('SERVER_IP', "0.0.0.0")
    server_port = os.getenv('SERVER_PORT', "8000")
    partition_id = int(os.getenv('PARTITION_ID', 0))
    num_partitions = int(os.getenv('NUM_PARTITIONS', 2))  # Default to 2 clients

    return server_ip, server_port, partition_id, num_partitions


if __name__ == "__main__":
    main()
