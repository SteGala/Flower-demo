from typing import List, Tuple
import os
import time

import flwr as fl
from flwr.common import Metrics
import requests

def send_duration(ep, duration):
    data = {"duration": duration}
    response = requests.post(f'http://{ep}/post_duration', json=data)
    if response.status_code == 200:
        print('Duration sent successfully')
    else:
        print('Error sending duration')

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class TimedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, metrics_server, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_server = metrics_server
        self.num_rounds = num_rounds
        self.start_time = None

    def aggregate_fit(self, rnd: int, results, failures):
        if self.start_time is None:
            self.start_time = time.time()
        
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if rnd == self.num_rounds:
            end_time = time.time()
            round_duration = end_time - self.start_time
            send_duration(self.metrics_server, round_duration)
        
        # print(f"Round {rnd} duration: {round_duration:.2f} seconds")
        
        # # Send round duration to metrics server
        # send_duration(self.metrics_server, round_duration)
        
        return aggregated_result

def read_env_variable():
    server_ip = os.getenv('SERVER_IP', "0.0.0.0")
    server_port = os.getenv('SERVER_PORT', "8000")
    num_rounds = int(os.getenv('NUM_ROUNDS', 5))
    metrics_server = os.getenv('METRICS_SERVER', "0.0.0.0:5000")
    return server_ip, server_port, num_rounds, metrics_server

if __name__ == "__main__":
    server_ip, server_port, num_rounds, metrics_server = read_env_variable()
    
    # Define custom strategy with timing
    strategy = TimedFedAvg(metrics_server=metrics_server, num_rounds=num_rounds, evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address=f"{server_ip}:{server_port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
