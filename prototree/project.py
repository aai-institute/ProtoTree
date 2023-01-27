from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import InternalNode, Node


# TODO: generalize to ProtoBase
@torch.no_grad()
def replace_prototypes_by_projections(
    tree: ProtoTree,
    project_loader: DataLoader,
    constrain_on_classes=True,
):
    """
    The goal is to find the latent patch that minimizes the L2 distance to each prototype.
    This is done by iterating through a dataset (typically the train dataset) and
    replacing each prototype by the closest latent patch.
    **Important**: This modifies the prototype weights in-place!

    :param tree:
    :param project_loader:
    :param constrain_on_classes: if True, only consider patches from classes that are contained in
        the prototype's leaves' predictions
    :return: a dictionary mapping nodes to an object holding information about the selected
        latent patch
    """
    node_to_patch_info: dict[Node, ProjectionPatchInfo] = {}

    @lru_cache(maxsize=1)
    def get_leaf_labels(node: InternalNode):
        return {leaf.predicted_label() for leaf in node.leaves}

    def process_sample(
        node: InternalNode,
        image_latent: torch.Tensor,
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
            node_to_patch_info[node] = ProjectionPatchInfo(
                node=node,
                image_latent=image_latent,
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

        for node in tree.internal_nodes:
            proto_idx = tree.node_to_proto_idx[node]

            for x_i, y_i, distances_i, patches_i in zip(
                x, y, distances[:, proto_idx, :, :], patches
            ):
                if constrain_on_classes and y_i.item() not in get_leaf_labels(node):
                    continue
                process_sample(node, x_i, y_i, distances_i, patches_i)

    def replace_proto_by_patch(proto_idx: int, patch: torch.Tensor):
        tree.prototype_layer.prototype_tensors.data[proto_idx] = patch.data

    for node, patch_info in node_to_patch_info.items():
        proto_idx = tree.node_to_proto_idx[node]
        replace_proto_by_patch(proto_idx, patch_info.closest_patch)
    return node_to_patch_info


@dataclass
class ProjectionPatchInfo:
    image_latent: torch.Tensor
    true_label: int
    closest_patch: torch.Tensor
    closest_patch_distance: float
    node: InternalNode
    all_patch_distances: torch.Tensor = None

    def get_similarities_latent(self) -> torch.Tensor:
        return torch.exp(-self.all_patch_distances)


def _get_closest_patch(sample_distances: torch.Tensor, sample_patches: torch.Tensor):
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
