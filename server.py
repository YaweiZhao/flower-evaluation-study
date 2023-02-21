import os
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import flwr as fl
from collections import OrderedDict
import torch
from network.DenseNet import DenseNet
from flwr.common import Metrics
import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

param = config.get_param()

DEVICE = param["bench_param"]["device"]
server_address = param["bench_param"]["server_address"]
num_rounds = param["bench_param"]["num_rounds"]
net = DenseNet().to(DEVICE)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            save_path = "./save_model"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(), os.path.join(save_path, f"model_round_{server_round}.pth"))
        
        return aggregated_parameters, aggregated_metrics



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    print("aggregated accuracies: ", sum(accuracies) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    # Create strategy and run server
    strategy = SaveModelStrategy(evaluate_metrics_aggregation_fn=weighted_average)
    
    fl.server.start_server(
    server_address = server_address,
    config=fl.server.ServerConfig(num_rounds),
    strategy=strategy,
)

if __name__ == "__main__":
    main()




