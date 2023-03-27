import torch
import torch.nn.functional as F


def min_pool2d(x:  torch.Tensor, kernel_size: torch.Size, **kwargs) -> torch.Tensor:
    """
    Applies 2D min pooling over an input signal composed of several input planes. This is based on `PyTorch's max
    pooling <https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html>`_.

    Args:
        x: Input tensor (minibatch, in_channels, iH, iW), minibatch dim optional.
        kernel_size: Size of the pooling region. Can be a single number or a tuple (kH, kW).
    """
    return -F.max_pool2d(-x, kernel_size, **kwargs)
