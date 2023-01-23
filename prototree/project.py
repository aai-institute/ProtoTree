from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import Node


def replace_prototypes_by_projections(
    tree: ProtoTree,
    project_loader: DataLoader,
    constrain_on_classes=True,
):
    """
    The goal is to find the latent patch that minimizes the L2 distance to each prototype.
    This is done by iterating through a dataset (typically the train dataset) and
    replacing each prototype by the closest latent patch.

    :param tree:
    :param project_loader:
    :param constrain_on_classes: if True, only consider patches from classes that are contained in
        the prototype's leaves' predictions
    :return:
    """
    tree.eval()
    # TODO: what for?
    torch.cuda.empty_cache()

    proto_idx_to_patch_dist = defaultdict(lambda: np.inf)
    w_proto, h_proto = tree.prototype_shape[:2]

    node_to_leaf_y_pred: dict[Node, set[int]] = {
        node: {leaf.predicted_label() for leaf in node.leaves}
        for node in tree.internal_nodes
    }

    with torch.no_grad():
        # TODO: is this the most efficient way of doing this? Maybe reverse loops or vectorize
        for x, y in tqdm(project_loader, desc="Projection", ncols=0):
            x, y = x.to(tree.device), y.to(tree.device)
            features = tree.extract_features(x)
            distances = tree.prototype_layer(features)
            # TODO -- support for strides in finding patches? (corresponds to step size = 1 here)
            # Shape: (batch_size, d, n_patches_w, n_patches_h, w_proto, h_proto)
            patches = features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

            for node in tree.internal_nodes:
                proto_idx = tree.node_to_proto_idx[node]
                leaf_predictions = node_to_leaf_y_pred[node]

                for y_i, distances_i, patches_i in zip(
                    y, distances[:, proto_idx, :, :], patches
                ):
                    if constrain_on_classes and y_i.item() not in leaf_predictions:
                        continue

                    closest_patch = _get_closest_patch(distances_i, patches_i)
                    closest_patch_distance = distances_i.min().item()

                    if closest_patch_distance < proto_idx_to_patch_dist[proto_idx]:
                        proto_idx_to_patch_dist[proto_idx] = closest_patch_distance
                        tree.prototype_layer.prototype_tensors.data[
                            proto_idx
                        ] = closest_patch.data


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
