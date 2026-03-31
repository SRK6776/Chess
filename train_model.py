import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# 1. Define the PyTorch Dataset wrapper
class ChessDataset(Dataset):
    def __init__(self, X_path, y_path):
        # Load the numpy arrays
        print("Loading data from disk...")
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        print(f"Loaded {len(self.X)} samples.")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to PyTorch tensors
        # X is already in (12, 8, 8) format which is perfect for PyTorch (Channels, Height, Width)
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# 2. Define the Convolutional Neural Network (CNN)
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        
        # Convolutions to learn spatial patterns on the 8x8 board
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Flatten the 128 channels of 8x8 boards into a 1D vector
        self.flatten = nn.Flatten()
        
        # Fully connected layers to map to the 4096 possible moves
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.relu4 = nn.ReLU()
        
        self.fc2 = nn.Linear(1024, 4096) # 4096 is 64 * 64 (start square * 64 + end square)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu4(x)
        
        x = self.fc2(x)
        return x

def train():
    # Setup device (macOS M-series usually supports MPS, otherwise CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 10

    # Load dataset
    full_dataset = ChessDataset('X.npy', 'y.npy')
    
    # Split 90% train / 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = ChessCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (boards, moves) in enumerate(train_loader):
            boards, moves = boards.to(device), moves.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(boards)
            loss = criterion(outputs, moves)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for boards, moves in val_loader:
                boards, moves = boards.to(device), moves.to(device)
                outputs = model(boards)
                loss = criterion(outputs, moves)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += moves.size(0)
                correct += (predicted == moves).sum().item()

        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100 * correct / total:.2f}%")
        print("---------------------------")

    # Save the trained model
    torch.save(model.state_dict(), "chess_model.pth")
    print("Model saved to chess_model.pth")

if __name__ == "__main__":
    # Ensure X.npy and y.npy exist before running
    if os.path.exists('X.npy') and os.path.exists('y.npy'):
        train()
    else:
        print("Error: X.npy and y.npy not found. Please run data_to_np.py first.")
