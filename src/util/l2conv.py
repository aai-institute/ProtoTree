import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Conv2D(nn.Module):
    """
    Convolutional layer that computes the squared L2 distances for each prototype
    instead of the conventional inner product.
    """

    def __init__(self, num_prototypes: int, input_channels: int, w: int, h: int, initial_mean=0.0, initial_std=1.0):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param input_channels: The number of channels in the input features
        :param w: Width of the prototypes
        :param h: Height of the prototypes
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.input_channels = input_channels
        self.w = w
        self.h = h
        self.prototype_shape = (w, h, input_channels)
        # TODO: make consistent ordering!!
        prototype_shape = (num_prototypes, input_channels, w, h)

        prototype_initial_values = torch.randn(*prototype_shape) * initial_std + initial_mean
        self.prototype_tensors = nn.Parameter(prototype_initial_values, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute the squared L2 distance for all prototypes and patches simultaneously by using
        convolutions.

        Returns a tensor of shape `(batch_size, num_prototypes, n_patches_w, n_patches_h)`
        obtained from computing the squared L2 distances for patches of the prototype shape from the input
        using all prototypes.

        Here `n_patches_w = W - w + 1` and `n_patches_h = H - h + 1` where `w` and `h` are the width and height of the
        prototypes. There are in total `n_patches_w * n_patches_h` patches of the prototype shape in the input.

        :param x: A batch of input images obtained as output from some convolutional neural network F. Following the
                   notation from the paper, the shape of x is `(batch_size, D, W, H)`.
        :return: a tensor of shape `(batch_size, num_prototypes, n_patches_w, n_patches_h)`.
        """
        # Adapted from ProtoPNet
        # ||xs - ps ||^2 = ||xs||^2 + ||ps||^2 - 2 * xs * ps

        # ||xs||^2 for all patches simultaneously by convolving with ones
        _ones = torch.ones_like(self.prototype_tensors, device=x.device)
        # Shape: (bs, num_prototypes, w_in - w + 1, h_in - h + 1)
        xs_squared_l2 = F.conv2d(x**2, weight=_ones)

        # Shape: (num_prototypes, )
        ps_squared_l2 = torch.sum(self.prototype_tensors**2, dim=(1, 2, 3))

        # Compute xs * ps for all patches simultaneously by convolving
        # Shape: (bs, num_prototypes, w_in, h_in)
        xs_conv = F.conv2d(x, weight=self.prototype_tensors)

        distances_sq = xs_squared_l2 - 2 * xs_conv + ps_squared_l2.view(-1, 1, 1)
        return torch.sqrt(distances_sq + 1e-14)
