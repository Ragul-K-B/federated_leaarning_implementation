import flwr as fl
import time
from flwr.server import start_server
from flwr.server.strategy import FedAvg

class CustomStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0

    def configure_fit(self, server_round: int, parameters: list, client_manager: fl.server.ClientManager):
        self.current_round = server_round
        return super().configure_fit(server_round, parameters, client_manager)

    def fit_round(self):
        # Custom round logic
        print(f"Starting custom round {self.current_round}")
        return super().fit_round()

def main():
    # Define the custom strategy
    strategy = CustomStrategy(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )
    
    # Define the server configuration
    config = fl.server.ServerConfig(
        num_rounds=10,  # Set the number of rounds
        round_timeout=None,  # No round timeout
    )
    
    # Start the Flower server with the custom strategy
    server = start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=config,
    )
    
    # Keep the server running indefinitely
    try:
        while True:
                server = start_server(
                    server_address="localhost:8080",
                    strategy=strategy,
                    config=config,
    )
                time.sleep(10)  # Sleep to keep the server running
    except KeyboardInterrupt:
        print("Server is stopping...")

if __name__ == "__main__":
    main()
