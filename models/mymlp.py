import torch
from torch import nn

# For reproducibility
torch.manual_seed(0)

class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout_rate=0.2):
        super(MyMLP, self).__init__()
        
        # Layer 1
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(self.linear1.weight)
        self.batch1 = nn.LayerNorm(hidden_dim)
        
        # Layer 2
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        torch.nn.init.kaiming_uniform_(self.linear2.weight)
        self.batch2 = nn.LayerNorm(hidden_dim//2)
        
        # Layer 3
        self.linear3 = nn.Linear(hidden_dim//2, hidden_dim//2)
        torch.nn.init.kaiming_uniform_(self.linear3.weight)
        self.batch3 = nn.LayerNorm(hidden_dim//2)

        # Output layer
        self.linear4 = nn.Linear(hidden_dim//2, output_dim)
        torch.nn.init.kaiming_uniform_(self.linear4.weight)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.linear2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.linear3(x)
        x = self.batch3(x)
        x = self.relu(x)

        out = self.linear4(x)
        return out

