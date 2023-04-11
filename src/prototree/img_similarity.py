import math
from dataclasses import dataclass

import torch

from prototree.node import InternalNode


@dataclass
class ImageProtoSimilarity:
    # TODO: Lots of data is denormalized into this. Would it be better to normalize it?
    # TODO: The methods on this class seem to be replicating logic from the tree/nodes, how do we avoid this?
    """
    Stores the similarities between each patch of an image and a node's prototype.
    """
    internal_node: InternalNode
    transformed_image: torch.Tensor  # The image (in non-latent space) after preliminary transformations.
    closest_patch: torch.Tensor
    closest_patch_distance: float
    all_patch_distances: torch.Tensor

    @property
    def all_patch_similarities(self) -> torch.Tensor:
        """
        The entries in the result are the similarity measure evaluated for each patch, i.e. the probabilities of being
        routed to the right node for each patch if that patch is the closest.
        Therefore, the largest values in the result correspond to the patches which most closely match the prototype.
        """
        return torch.exp(-self.all_patch_distances)

    @property
    def highest_patch_similarity(self) -> float:
        """
        The similarity measure for the closest patch, i.e. the probability of being routed to the right node.
        """
        return math.exp(-self.closest_patch_distance)


@torch.no_grad()
def img_proto_similarity(
    internal_node: InternalNode,
    transformed_image: torch.Tensor,
    sample_patches_distances: torch.Tensor,
    sample_patches: torch.Tensor,
) -> ImageProtoSimilarity:
    # TODO: Doesn't the PrototypeBase kind of do this already? Why do we need to do further calculation here instead of
    #  lightly modifying PrototypeBase?
    """
    Calculates [node prototype]-[image] similarity (ImageProtoSimilarity) for a single (node, image) pair.
    """
    closest_patch = _get_closest_patch(sample_patches_distances, sample_patches)
    closest_patch_distance = sample_patches_distances.min().item()

    return ImageProtoSimilarity(
        internal_node=internal_node,
        transformed_image=transformed_image,
        closest_patch=closest_patch,
        closest_patch_distance=closest_patch_distance,
        all_patch_distances=sample_patches_distances,
    )


def _get_closest_patch(
    sample_distances: torch.Tensor, sample_patches: torch.Tensor
) -> torch.Tensor:
    # TODO: Doesn't the PrototypeBase kind of do this already? Why do we need lots of extra code here instead of lightly
    #  modifying PrototypeBase?
    """
    Get the closest latent patch based on the distances to the prototype. This is just a helper
    function for dealing with multidimensional indices.

    :param sample_distances: tensor of shape (n_patches_w, n_patches_h),
        representing the distances of the latent patches to a single prototype.
        Typically, the output of the prototype layer, see `ProtoTree.get_similarities` .
    :param sample_patches: tensor of shape (d_features, n_patches_w, n_patches_h, w_proto, h_proto),
        representing the latent patches of a single sample. Note that d_features equals the number of
        input channels in the prototype layer.
        Typically, the output of the feature extractor, see `ProtoTree.extract_features` .
    :return: tensor of shape (d_features, w_proto, h_proto), representing the closest latent patch
    """
    # Index in a flattened array of the patches/feature-map
    min_distance_ix = sample_distances.argmin().item()

    d_proto, n_patches_w, n_patches_h, w_proto, h_proto = sample_patches.shape
    n_patches = n_patches_w * n_patches_h
    # the index now runs in range(n_patches)
    patches_with_flat_index = sample_patches.reshape(
        d_proto, n_patches, w_proto, h_proto
    )
    return patches_with_flat_index[:, min_distance_ix, :, :]
