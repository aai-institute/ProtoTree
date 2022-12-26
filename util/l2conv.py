import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Conv2D(nn.Module):

    """
    Convolutional layer that computes the squared L2 distance instead of the conventional inner product.
    """

    def __init__(self, num_prototypes: int, input_channels: int, w1: int, h1: int):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param input_channels: The number of channels in the input features
        :param w1: Width of the prototypes
        :param h1: Height of the prototypes
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_shape = (w1, h1, input_channels)
        # TODO: make consistent ordering!!
        prototype_shape = (num_prototypes, input_channels, w1, h1)
        self.prototype_vectors = nn.Parameter(
            torch.randn(prototype_shape), requires_grad=True
        )

    def forward(self, xs):
        """
        Perform convolution over the input using the squared L2 distance for all prototypes in the layer
        :param xs: A batch of input images obtained as output from some convolutional neural network F. Following the
                   notation from the paper, let the shape of xs be (batch_size, D, W, H), where
                     - D is the number of output channels of the conv net F
                     - W is the width of the convolutional output of F
                     - H is the height of the convolutional output of F
        :return: a tensor of shape (batch_size, num_prototypes, W, H) obtained from computing the squared L2 distances
                 for patches of the input using all prototypes
        """
        # Adapted from ProtoPNet
        # Computing ||xs - ps ||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
        # where ps is some prototype image

        # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
        # with weights set to 1 so each patch just has its values summed)
        # Shape: (num_prototypes, num_features, w_1, h_1)
        ones = torch.ones_like(self.prototype_vectors, device=xs.device)
        # Shape: (bs, num_prototypes, w_in, h_in)
        xs_squared_l2 = F.conv2d(xs**2, weight=ones)

        # Now compute ||ps||^2
        # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
        # squared L2 distance
        # Shape: (num_prototypes,)
        ps_squared_l2 = torch.sum(self.prototype_vectors**2, dim=(1, 2, 3))
        # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)

        # Compute xs * ps (for all patches in the input image)
        # Shape: (bs, num_prototypes, w_in, h_in)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)

        # Use the values to compute the squared L2 distance
        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        # L2 distance (not squared). Small epsilon added for numerical stability
        distance = torch.sqrt(torch.abs(distance) + 1e-14)

        if torch.isnan(distance).any():
            # TODO: improve error handling, this should not happen at all
            raise RuntimeError(
                "Error: NaN values! Using the --log_probabilities flag might fix this issue"
            )
        # Shape: (bs, num_prototypes, w_in, h_in)
        return distance
