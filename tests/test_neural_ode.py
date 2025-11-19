import pytest
import torch
import numpy as np
from src.ml.neural_ode import NeuralODE

def test_neural_ode_initialization():
    model = NeuralODE(data_dim=2)
    assert isinstance(model, torch.nn.Module)

def test_neural_ode_forward():
    # This test requires torchdiffeq. If not present, it should fail gracefully or skip.
    try:
        import torchdiffeq
    except ImportError:
        pytest.skip("torchdiffeq not installed")

    model = NeuralODE(data_dim=2)
    y0 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0, 1, 10)
    
    # Forward pass
    out = model(y0, t)
    
    # Check shape: (n_times, batch_size, data_dim) -> (10, 1, 2)
    assert out.shape == (10, 1, 2)
