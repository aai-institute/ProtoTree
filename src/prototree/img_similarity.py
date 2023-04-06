from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import InternalNode


@dataclass
class ImageProtoSimilarity:
    """
    Stores the similarities between each patch of an image and a node's prototype.
    TODO: Lots of data is denormalized into this. Perhaps it should be normalized.
    """

    internal_node: InternalNode
    transformed_image: torch.Tensor  # The image (in non-latent space) after preliminary transformations.
    closest_patch: torch.Tensor
    closest_patch_distance: float
    all_patch_distances: torch.Tensor

    def all_patch_similarities(self) -> torch.Tensor:
        """
        The entries in the result are the similarity measure evaluated for each patch, i.e. the probabilities of being
        routed to the right node for each patch.
        Therefore, the largest values in the result correspond to the patches which most closely match the prototype.
        """
        return torch.exp(-self.all_patch_distances)


# TODO: Generalize to ProtoBase
@torch.no_grad()
def calc_node_patch_matches(
    tree: ProtoTree,
    loader: DataLoader,
    constrain_on_classes=True,
):
    """
    Produces a map where each key is a node and the corresponding value is information about the patch (out of all
    images in the dataset) that is most similar to node's prototype.

    :param tree:
    :param loader: The dataset.
    :param constrain_on_classes: If True, only consider patches from classes that are contained in
        the prototype's leaves' predictions.
    :return The map of nodes to best matches.
    """

    # TODO: Should this be a method on the node? If this weren't an inner function, we'd need to beware of caching
    #  incorrect results when the leaf logits change.
    @lru_cache(maxsize=10000)
    def get_leaf_labels(internal_node: InternalNode):
        return {leaf.predicted_label() for leaf in internal_node.leaves}

    # TODO: Is there a more functional way of doing this?
    node_to_patch_matches: dict[["InternalNode"], ImageProtoSimilarity] = {}
    for proto_similarity, label in patch_match_candidates(tree, loader):
        if (not constrain_on_classes) or label in get_leaf_labels(
            proto_similarity.internal_node
        ):
            node = proto_similarity.internal_node
            cur_closest = node_to_patch_matches[node]
            if (
                (node not in node_to_patch_matches)
                or proto_similarity.closest_patch_distance
                < cur_closest.closest_patch_distance
            ):
                node_to_patch_matches[node] = proto_similarity

    return node_to_patch_matches


# TODO: Lots of overlap with Prototree.justify, but we need to beware of premature abstraction.
@torch.no_grad()
def patch_match_candidates(
    tree: ProtoTree, loader: DataLoader
) -> Iterator[(ImageProtoSimilarity, int)]:
    """
    Generator yielding the [node prototype]-[image] similarity (ImageProtoSimilarity) for every (node, image) pair in
    the given tree and dataloader. A generator is used to avoid OOMing on larger datasets and trees.

    Returns: Iterator of (similarity, label)
    """
    for x, y in tqdm(loader, desc="Data loader", ncols=0):
        x, y = x.to(tree.device), y.to(tree.device)
        patches, distances = tree.patches(x), tree.distances(
            x
        )  # Could be optimized if necessary.

        for x_i, y_i, distances_i, patches_i in zip(
                x, y, distances, patches
        ):
            for internal_node in tree.internal_nodes:
                node_proto_idx = tree.node_to_proto_idx[internal_node]

                node_distances = distances_i[node_proto_idx, :, :]
                similarity = img_proto_similarity(
                    internal_node, x_i, distances_i, patches_i
                )
                yield similarity, y_i


@torch.no_grad()
def img_proto_similarity(
    internal_node: InternalNode,
    transformed_image: torch.Tensor,
    sample_patches_distances: torch.Tensor,
    sample_patches: torch.Tensor,
) -> ImageProtoSimilarity:
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
