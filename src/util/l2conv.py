from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Conv2D(nn.Module):
    """
    Convolutional layer that computes the squared L2 distances for each prototype
    instead of the conventional inner product.
    """

    def __init__(
            self,
            num_prototypes: int,
            input_channels: int,
            w: int,
            h: int,
            initial_mean=0.0,
            initial_std=1.0,
    ):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param input_channels: The number of channels in the input features
        :param w: Width of the prototypes
        :param h: Height of the prototypes
        :param initial_mean: Initialize the prototypes with a Gaussian with this mean.
        :param initial_std: Initialize the prototypes with a Gaussian with this standard deviation.
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.input_channels = input_channels
        self.w = w
        self.h = h
        self.prototype_shape = (w, h, input_channels)
        # TODO: make consistent ordering!!
        prototype_shape = (num_prototypes, input_channels, w, h)

        prototype_initial_values = (
                torch.randn(*prototype_shape) * initial_std + initial_mean
        )
        self.prototype_tensors = nn.Parameter(
            prototype_initial_values, requires_grad=True
        )

    def forward(self, x: torch.Tensor, proto_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if proto_indices is None:
            prototypes = self.prototype_tensors
        else:
            # TODO: This is really inefficient, we're computing the distance for every (image i, prototypes for image j)
            #  pair in the batch and then taking the diagonal where i == j. If this becomes a bottleneck we should do it
            #  properly.
            proto_indices_flat = torch.flatten(proto_indices)
            prototypes = self.prototype_tensors[proto_indices_flat, ...]

        dists = L2Conv2D._prototype_dists(x, prototypes)

        if proto_indices is None:
            return dists
        else:
            # TODO: Check this reshaping.
            unflattened_shape = (dists.shape[0], *proto_indices.shape, *dists.shape[2:])
            unflattened_dists = torch.reshape(dists, unflattened_shape)
            dists_for_imgs = torch.diagonal(unflattened_dists, dim1=0, dim2=1)
            permutation = deque(range(len(dists_for_imgs.shape)))
            permutation.rotate()
            return torch.permute(dists_for_imgs, tuple(permutation))

    @staticmethod
    def _prototype_dists(x: torch.Tensor, prototypes: torch.Tensor):
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
        _ones = torch.ones_like(prototypes)
        # Shape: (bs, num_prototypes, w_in - w + 1, h_in - h + 1)
        xs_squared_l2 = F.conv2d(x ** 2, weight=_ones)

        # Shape: (num_prototypes, )
        ps_squared_l2 = torch.sum(prototypes ** 2, dim=(1, 2, 3))

        # Compute xs * ps for all patches simultaneously by convolving
        # Shape: (bs, num_prototypes, w_in, h_in)
        xs_conv = F.conv2d(x, weight=prototypes)

        # TODO: Negative numbers can appear here, so we have to clamp to an epsilon. Adding the epsilon is a viable
        #  alternative, but it would need to be quite large to handle the negative numbers we're seeing,
        #  e.g. eps >> 1e-7.
        #      1. Figure out why seemingly insignificant changes (e.g. refactors with no intended change to the
        #         numerical calculations, smaller batches, bumping library versions) cause negative numbers to start
        #         appearing, or appear more quickly.
        #      2. What numerical instabilities are causing negative numbers that are so far below 0?
        #      3. Can't we just compute ||xs - ps ||^2 directly, instead of expanding to
        #         ||xs||^2 + ||ps||^2 - 2 * xs * ps ? Is there a vectorized way to do it without creating bigger
        #         tensors?
        distances_sq = xs_squared_l2 - 2 * xs_conv + ps_squared_l2.view(-1, 1, 1)
        distances_sq_clamped = torch.clamp(distances_sq, min=1e-14)

        return torch.sqrt(distances_sq_clamped)  # TODO: Pick good eps.
