import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint, uniform
from logging import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, X, y, scaler=None):
        if scaler:
            self.X = torch.FloatTensor(scaler.transform(X))
        else:
            self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    """Neural Network Architecture"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(NeuralNetwork, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NeuralNetworkTrainer:
    def __init__(self, logger: Logger, results_path: str):
        """
        Initialize NeuralNetworkTrainer
        
        Args:
            logger (Logger): Logger instance for tracking training process
            results_path (str): Path to save training results and parameters
        """
        self.logger = logger
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def get_model_dir(self, position: str = None, stat_type: str = None) -> str:
        """Get the directory path for a specific model"""
        if position and stat_type:
            return os.path.join(self.results_path, 'neural_network_results', f'{position}_{stat_type}')
        return os.path.join(self.results_path, 'neural_network_results')
        
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, 
              position: str = None, stat_type: str = None,
              batch_size: int = 32, num_epochs: int = 100, 
              learning_rate: float = 0.001) -> tuple:
        """
        Train Neural Network model with position/stat-type specific storage
        
        Args:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            position (str): Player position (QB, RB, etc.)
            stat_type (str): Type of stat (passing, rushing, etc.)
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
            learning_rate (float): Initial learning rate
        """
        self.logger.info(f"Training Neural Network Model for {position} {stat_type}")
        
        # Fit and transform the input features
        x_scaled = self.scaler.fit_transform(x_train)
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(x_train, y_train, self.scaler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Initialize model
        input_size = x_train.shape[1]
        model = NeuralNetwork(input_size).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5, 
                                                       verbose=True)
        
        # Training loop
        best_loss = float('inf')
        best_model_state = None
        train_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save training results and scaler
        self._save_training_results(model, train_losses, position, stat_type)
        
        return model
    
    def predict(self, model: nn.Module, X: pd.DataFrame, 
                position: str = None, stat_type: str = None) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            model (nn.Module): Trained neural network model
            X (pd.DataFrame): Input features
            position (str): Player position
            stat_type (str): Type of stat
        """
        model.eval()
        
        # Load the scaler if not already in memory
        if not hasattr(self, 'scaler') or not hasattr(self.scaler, 'mean_'):
            model_dir = self.get_model_dir(position, stat_type)
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def _save_training_results(self, model: nn.Module, train_losses: list,
                             position: str = None, stat_type: str = None):
        """
        Save training results, model parameters, and scaler
        
        Args:
            model (nn.Module): Trained model
            train_losses (list): Training loss history
            position (str): Player position
            stat_type (str): Type of stat
        """
        model_dir = self.get_model_dir(position, stat_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_state.pth'))
        
        # Save scaler using pickle
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training losses
        pd.DataFrame({'loss': train_losses}).to_csv(
            os.path.join(model_dir, 'training_losses.csv'), index=False)
        
        # Save model architecture
        with open(os.path.join(model_dir, 'model_architecture.txt'), 'w') as f:
            f.write(str(model))