import argparse
from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import Node
from util.log import Log

# TODO: this is unused, delete?
# def project(
#     tree: ProtoTree,
#     project_loader: DataLoader,
#     device,
#     log: Log,
#     progress_prefix: str = "Projection",
# ) -> tuple[ProtoTree, dict]:
#
#     log.log_message(
#         "\nProjecting prototypes to nearest training patch (without class restrictions)..."
#     )
#     # Set the model to evaluation mode
#     tree.eval()
#     torch.cuda.empty_cache()
#     # The goal is to find the latent patch that minimizes the L2 distance to each prototype
#     # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
#     # Also store info about the image that was used for projection
#     global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
#     global_min_patches = {j: None for j in range(tree.num_prototypes)}
#     global_min_info = {j: None for j in range(tree.num_prototypes)}
#
#     # Get the shape of the prototypes
#     W1, H1, D = tree.prototype_shape
#
#     # Build a progress bar for showing the status
#     projection_iter = tqdm(
#         enumerate(project_loader),
#         total=len(project_loader),
#         desc=progress_prefix,
#         ncols=0,
#     )
#
#     with torch.no_grad():
#         # Get a batch of data
#         xs, ys = next(iter(project_loader))
#         batch_size = xs.shape[0]
#         for i, (xs, ys) in projection_iter:
#             xs, ys = xs.to(device), ys.to(device)
#             # Get the features and distances
#             # - features_batch: features tensor (shared by all prototypes)
#             #   shape: (batch_size, D, W, H)
#             # - distances_batch: distances tensor (for all prototypes)
#             #   shape: (batch_size, num_prototypes, W, H)
#             # - out_map: a dict mapping decision nodes to distances (indices)
#             features_batch, distances_batch, out_map = tree.forward_partial(xs)
#
#             # Get the features dimensions
#             bs, D, W, H = features_batch.shape
#
#             # Get a tensor containing the individual latent patches
#             # Create the patches by unfolding over both the W and H dimensions
#             # TODO -- support for strides in the prototype layer? (corresponds to step size here)
#             patches_batch = features_batch.unfold(2, W1, 1).unfold(
#                 3, H1, 1
#             )  # Shape: (batch_size, D, W, H, W1, H1)
#
#             # Iterate over all decision nodes/prototypes
#             for node, j in out_map.items():
#
#                 # Iterate over all items in the batch
#                 # Select the features/distances that are relevant to this prototype
#                 # - distances: distances of the prototype to the latent patches
#                 #   shape: (W, H)
#                 # - patches: latent patches
#                 #   shape: (D, W, H, W1, H1)
#                 for batch_i, (distances, patches) in enumerate(
#                     zip(distances_batch[:, j, :, :], patches_batch)
#                 ):
#
#                     # Find the index of the latent patch that is closest to the prototype
#                     min_distance = distances.min()
#                     min_distance_ix = distances.argmin()
#                     # Use the index to get the closest latent patch
#                     closest_patch = patches.view(D, W * H, W1, H1)[
#                         :, min_distance_ix, :, :
#                     ]
#
#                     # Check if the latent patch is closest for all data samples seen so far
#                     if min_distance < global_min_proto_dist[j]:
#                         global_min_proto_dist[j] = min_distance
#                         global_min_patches[j] = closest_patch
#                         global_min_info[j] = {
#                             "input_image_ix": i * batch_size + batch_i,
#                             "patch_ix": min_distance_ix.item(),  # Index in a flattened array of the feature map
#                             "W": W,
#                             "H": H,
#                             "W1": W1,
#                             "H1": H1,
#                             "distance": min_distance.item(),
#                             "nearest_input": torch.unsqueeze(xs[batch_i], 0),
#                             "node_ix": node.index,
#                         }
#
#             # Update the progress bar if required
#             projection_iter.set_postfix_str(f"Batch: {i + 1}/{len(project_loader)}")
#
#             del features_batch
#             del distances_batch
#             del out_map
#         # Copy the patches to the prototype layer weights
#         projection = torch.cat(
#             tuple(
#                 global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)
#             ),
#             dim=0,
#             out=tree.prototype_layer.prototype_vectors,
#         )
#         del projection
#
#     return tree, global_min_info


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
    # TODO: what for?
    tree.eval()
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
