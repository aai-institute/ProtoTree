from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import InternalNode, Node


@dataclass
class ImageProtoSimilarity:
    """
    Stores the similarities between each patch of an image and a prototype.
    TODO: This stores the data in a partly denormalized manner since it holds the image, label, and patch. We should
     probably make it fully normalized.
    """
    transformed_image: torch.Tensor  # The image (in non-latent space) after preliminary transformations.
    true_label: int
    closest_patch: torch.Tensor
    closest_patch_distance: float
    all_patch_distances: torch.Tensor

    def get_similarities_latent(self) -> torch.Tensor:
        """
        The entries in the result are the similarity measure evaluated for each patch, i.e. the probabilities of being
        routed to the right node for each patch.
        Therefore, the largest values in the result correspond to the patches which most closely match the prototype.
        """
        return torch.exp(-self.all_patch_distances)


def calc_proto_similarity(
    transformed_image: torch.Tensor,
    true_label: int,
    sample_patches_distances: torch.Tensor,
    sample_patches: torch.Tensor,
) -> ImageProtoSimilarity:
    closest_patch = _get_closest_patch(sample_patches_distances, sample_patches)
    closest_patch_distance = sample_patches_distances.min().item()

    return ImageProtoSimilarity(
            transformed_image=transformed_image,
            true_label=true_label,
            closest_patch=closest_patch,
            closest_patch_distance=closest_patch_distance,
            all_patch_distances=sample_patches_distances,
        )


# TODO: generalize to ProtoBase
@torch.no_grad()
def calc_node_patch_info(
    tree: ProtoTree,
    project_loader: DataLoader,
    constrain_on_classes=True,
):
    """
    Produces a map containing information about the similarity between prototypes and the patches.

    :param tree:
    :param project_loader:
    :param constrain_on_classes: if True, only consider patches from classes that are contained in
        the prototype's leaves' predictions
    :return: a dictionary mapping nodes to an object holding information about the selected
        latent patch
    """
    node_to_patch_info: dict[Node, ImageProtoSimilarity] = {}

    @lru_cache(maxsize=1)
    def get_leaf_labels(node: InternalNode):
        return {leaf.predicted_label() for leaf in node.leaves}


    def process_sample(
        node: InternalNode,
        transformed_image: torch.Tensor,
        true_label: int,
        sample_patches_distances: torch.Tensor,
        sample_patches: torch.Tensor,
    ):
        prev_patch_info = node_to_patch_info.get(node)

        closest_patch = _get_closest_patch(sample_patches_distances, sample_patches)
        closest_patch_distance = sample_patches_distances.min().item()

        if (
            not prev_patch_info
            or closest_patch_distance < prev_patch_info.closest_patch_distance
        ):
            node_to_patch_info[node] = ImageProtoSimilarity(
                transformed_image=transformed_image,
                true_label=true_label,
                closest_patch=closest_patch,
                closest_patch_distance=closest_patch_distance,
                all_patch_distances=sample_patches_distances,
            )

    # TODO: is this the most efficient way of doing this? Maybe reverse loops or vectorize
    # The logic is: loop over batches -> loop over nodes ->
    # loop over samples in batch to find closest patch for current node
    w_proto, h_proto = tree.prototype_shape[:2]
    for x, y in tqdm(project_loader, desc="Projection", ncols=0):
        x, y = x.to(tree.device), y.to(tree.device)
        features = tree.extract_features(x)
        distances = tree.prototype_layer(features)
        # Shape: (batch_size, d, n_patches_w, n_patches_h, w_proto, h_proto)
        patches = features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

        for internal_node in tree.internal_nodes:
            node_proto_idx = tree.node_to_proto_idx[internal_node]

            for x_i, y_i, distances_i, patches_i in zip(
                x, y, distances[:, node_proto_idx, :, :], patches
            ):
                if constrain_on_classes and y_i.item() not in get_leaf_labels(internal_node):
                    continue
                process_sample(internal_node, x_i, y_i, distances_i, patches_i)

    return node_to_patch_info


@torch.no_grad()
def replace_prototypes_with_patches(tree: ProtoTree, node_to_patch_info: dict[Node, ImageProtoSimilarity]):
    """
    Replaces each prototype with a given patch.
    Note: This mutates the prototype tensors.
    TODO: We should probably not be mutating the tree (via the prototypes), as this is making the code less flexible and
     harder to reason about.
    """
    for internal_node, patch_info in node_to_patch_info.items():
        node_proto_idx = tree.node_to_proto_idx[internal_node]
        tree.prototype_layer.prototype_tensors.data[node_proto_idx] = patch_info.closest_patch.data


def _get_closest_patch(sample_distances: torch.Tensor, sample_patches: torch.Tensor) -> torch.Tensor:
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
