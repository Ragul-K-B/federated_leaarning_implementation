import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_data(file_path, input_dim):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Prepare the data (assuming 'target' is the column to predict)
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(1)  # Ensure target has shape (N, 1)

    # Create TensorDataset
    dataset = TensorDataset(features_tensor, target_tensor)
    
    # Create DataLoader instances
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader
