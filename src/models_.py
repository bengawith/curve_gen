import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Any


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, activation_func: Any = None, dropout_rate: float = 0.0, batch_norm: bool = False):
        super(FullyConnectedNN, self).__init__()
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            in_size = input_size if i == 0 else hidden_sizes[i - 1]
            layers.append(nn.Linear(in_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU() if activation_func is None else activation_func)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """
    A Recurrent Neural Network using GRU layers.
    
    This model expects input as a sequence. If a 2D input is provided, it is unsqueezed to have a sequence length of 1.
    
    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of features in the hidden state.
        output_size (int): Number of output features.
        num_layers (int): Number of stacked GRU layers.
        dropout (float): Dropout rate applied between GRU layers.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.0):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    
def adam(model: torch.nn.Module, learning_rate: float) -> torch.optim.Adam:
    return Adam(model.parameters(), lr=learning_rate)


def instantiate_model(net_name: str, input_size: int, output_size: int, params: dict, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    if net_name == "FullyConnectedNN":
        num_layers = params["num_layers"]
        hidden_sizes = [params[f"hidden_size_{i}"] for i in range(num_layers)]
        dropout_rate = params["dropout_rate"]
        model = FullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, dropout_rate=dropout_rate).to(device)

    elif net_name == "LSTMModel":
        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        dropout_rate = params["dropout_rate"]
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                          num_layers=num_layers, dropout=dropout_rate).to(device)
        
    elif net_name == "GRUModel":
        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        dropout_rate = params["dropout_rate"]
        model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                         num_layers=num_layers, dropout=dropout_rate).to(device)
        
    else:
        raise ValueError(f"Unknown network type: {net_name}")
    return model