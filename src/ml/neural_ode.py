"""
Neural ODE implementation using PyTorch.
"""
import torch
import torch.nn as nn

class NeuralODEFunc(nn.Module):
    def __init__(self, data_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, data_dim, hidden_dim=64, solver='dopri5'):
        super().__init__()
        self.func = NeuralODEFunc(data_dim, hidden_dim)
        self.solver = solver

    def forward(self, y0, t):
        try:
            from torchdiffeq import odeint
            return odeint(self.func, y0, t, method=self.solver)
        except ImportError:
            raise ImportError("torchdiffeq is required for NeuralODE. Please install it.")
