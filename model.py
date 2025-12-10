import torch
import torch.nn as nn

class StatPredictor(nn.Module):
    def __init__(self, input_size=18, output_size=7):
        super(StatPredictor, self).__init__()
        
        self.net = nn.Sequential(
            # Input -> Hidden
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden -> Hidden
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Hidden -> Output
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.net(x)