import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_size))
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)