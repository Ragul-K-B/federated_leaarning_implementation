import flwr as fl
import torch
import numpy as np
from model import Model
from dataloader import load_data

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, file_path, input_dim):
        self.model = Model(input_dim=input_dim)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader, self.test_loader = load_data(file_path, input_dim)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Adjust number of epochs as needed
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        # Return the required tuple with updated model parameters, number of samples, and loss
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}

    def get_feature_columns(self):
        feature_columns = [
            "age", "sex", "cp", "trestbps", "chol", 
            "fbs", "restecg", "thalach", "exang", 
            "oldpeak", "slope", "ca", "thal"
        ]
        return feature_columns

    def predict(self, user_input):
        self.model.eval()
        user_input = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(user_input)
        return output.item()

def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Client for Heart Disease Prediction.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimension of the input features.')
    
    args = parser.parse_args()
    
    # Retrieve arguments
    file_path = args.file_path
    input_dim = args.input_dim
    
    client = HeartDiseaseClient(file_path, input_dim)
    
    # Print feature columns
    print("Feature Columns:", client.get_feature_columns())

    # Example user input for prediction
    age = float(input("ENTER AGE: "))
    sex = float(input("ENTER Sex: "))
    cp = float(input("ENTER cp: "))
    trestbps = float(input("ENTER trestbps: "))
    chol = float(input("ENTER chol: "))
    fbs = float(input("ENTER fbs: "))
    restecg = float(input("ENTER restecg: "))
    thalach = float(input("ENTER thalach: "))
    exang = float(input("ENTER exang: "))
    oldpeak = float(input("ENTER oldpeak: "))
    slope = float(input("ENTER slope: "))
    ca = float(input("ENTER ca: "))
    thal = float(input("ENTER thal: "))

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Predict using the client model
    prediction = client.predict(user_input)
    print(f"Prediction: {prediction}")

    # Start the federated learning client
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
