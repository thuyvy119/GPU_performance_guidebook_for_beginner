import torch

def row_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=1)