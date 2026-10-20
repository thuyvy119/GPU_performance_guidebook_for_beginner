import torch

def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)