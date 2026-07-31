import os
import flwr as fl
import torch
from flwr.server.strategy import FedAvg
from common import load_model, load_tokenizer

class SaveStrategy(FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            model = load_model()

            params = fl.common.parameters_to_ndarrays(aggregated_parameters)

            trainable = [p for p in model.parameters() if p.requires_grad]

            # Fix 1: Explicit device placement matching p.device
            for p, nd in zip(trainable, params):
                p.data = torch.tensor(nd, dtype=p.dtype, device=p.device)

            # Fix 2: Create subfolder per round so checkpoints aren't overwritten
            save_dir = f"saved_global/round_{server_round}"
            os.makedirs(save_dir, exist_ok=True)

            model.save_pretrained(save_dir)

            tokenizer = load_tokenizer()
            tokenizer.save_pretrained(save_dir)

            print(f" Saved global model checkpoint after round {server_round} to '{save_dir}'")

        return aggregated_parameters, metrics

# Fix 3: Main guard wrapper
if __name__ == "__main__":
    strategy = SaveStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3
    )

    print(" Starting Federated Server on 0.0.0.0:8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )