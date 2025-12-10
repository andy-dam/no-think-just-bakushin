import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import AoharuDataset
from model import StatPredictor

# --- CONFIGURATION ---
CSV_FILE = 'training_data.csv'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
SAVE_PATH = "aoharu_model.pth"

def train_model():
    # 1. Prepare Data
    try:
        dataset = AoharuDataset(CSV_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    model = StatPredictor()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Standard for Regression

    print(f"Starting training on {len(dataset)} examples...")
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation logging
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(test_loader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_model()