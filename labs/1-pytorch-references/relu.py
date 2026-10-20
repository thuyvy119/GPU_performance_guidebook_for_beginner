import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0)
