"""Flower server example."""

from typing import List, Tuple
import os

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def read_env_variable():
    # get node ID
    server_ip = os.getenv('SERVER_IP', "0.0.0.0")
    server_port = os.getenv('SERVER_PORT', "8000")
    num_rounds = int(os.getenv('NUM_ROUNDS', 5))

    return server_ip, server_port, num_rounds

server_ip, server_port, num_rounds = read_env_variable()

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address=f"{server_ip}:{server_port}",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
