import argparse
from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.prototree import ProtoTree
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


def project_with_class_constraints(
    tree: ProtoTree,
    project_loader: DataLoader,
    log: Log,
    progress_prefix: str = "Projection",
) -> tuple[ProtoTree, dict]:

    log.log_message(
        "\nProjecting prototypes to nearest training patch (with class restrictions)..."
    )
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    w_proto, h_proto = tree.prototype_shape[:2]

    # Build a progress bar for showing the status
    projection_iter = tqdm(
        enumerate(project_loader),
        total=len(project_loader),
        desc=progress_prefix,
        ncols=0,
    )

    # TODO: 6-8 levels of for if else, I want to carve my eyes out
    with torch.no_grad():
        # Get a batch of data
        x, y = next(iter(project_loader))
        batch_size = x.shape[0]
        # For each internal node, collect the leaf labels in the subtree with this node as root.
        # Only images from these classes can be used for projection.
        # TODO: move to node method or delete
        node_id_to_leaf_predicted_labels: dict[int, set[int]] = {}

        # TODO 1: this uses out_map instead of relying on nodes. Has strange interferences with prune.
        #   I think prune doesn't modify the out_map. All out_map related code should be removed!
        for node in tree.out_map:
            node_id_to_leaf_predicted_labels[node.index] = {
                leaf.predicted_label() for leaf in node.leaves
            }

        for i, (x, y) in projection_iter:
            x, y = x.to(tree.device), y.to(tree.device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features = tree.extract_features(x)

            distances = tree.prototype_layer(features)

            # w, h = features.shape[-2:]

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            # Shape: (batch_size, D, W, H, W1, H1)
            patches = features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

            # Iterate over all decision nodes/prototypes
            for node, j in tree.out_map.items():
                leaf_labels = node_id_to_leaf_predicted_labels[node.index]
                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)

                # this iterates over the batch, thus each item corresponds to one sample
                for sample_i, (sample_distances, sample_patches) in enumerate(
                    zip(distances[:, j, :, :], patches)
                ):
                    # TODO: this is ugly, there is no else. Should adjust filtering logic
                    # Check if label of this image is in one of the leaves of the subtree
                    if y[sample_i].item() in leaf_labels:
                        closest_patch = _get_closest_patch(
                            sample_distances, sample_patches
                        )

                        # Check if the latent patch is closest for all data samples seen so far
                        min_distance = sample_distances.min().item()
                        if min_distance < global_min_proto_dist[j]:
                            global_min_proto_dist[j] = min_distance
                            global_min_patches[j] = closest_patch
                            global_min_info[j] = {
                                "input_image_ix": i * batch_size + sample_i,
                                # "patch_ix": min_distance_ix,
                                # "W": w,
                                # "H": h,
                                # "W1": W1,
                                # "H1": H1,
                                "distance": min_distance,
                                "nearest_input": torch.unsqueeze(x[sample_i], 0),
                                "node_ix": node.index,
                            }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f"Batch: {i + 1}/{len(project_loader)}")

            # TODO: why the del? Commented out for now
            # del features_batch
            # del distances_batch
            # del out_map

        # Copy the patches to the prototype layer weights
        # TODO: the fuck is this? I commented it out
        # projection = torch.cat(
        #     tuple(
        #         global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)
        #     ),
        #     dim=0,
        #     out=tree.prototype_layer.prototype_vectors,
        # )
        # del projection

    return tree, global_min_info


def _get_closest_patch(sample_distances: torch.Tensor, sample_patches: torch.Tensor):
    """
    Get the closest latent patch based on the distances to the prototype. This is just a helper
    function for dealing with multidimensional indices.

    :param sample_distances: tensor of shape (n_patches_w, n_patches_h),
        representing the distances of the latent patches to a single prototype.
    :param sample_patches: tensor of shape (d_features, n_patches_w, n_patches_h, w_proto, h_proto),
        representing the latent patches of a single sample. Note that d_features equals the number of
        input channels in the prototype layer.
    :return:
    """
    # Index in a flattened array of the patches/feature-map
    min_distance_ix = sample_distances.argmin().item()

    d_proto, n_patches_w, n_patches_h, w_proto, h_proto = sample_patches.shape
    n_patches = n_patches_w * n_patches_h
    # the index now runs in range(n_patches)
    patches_with_flat_index = sample_patches.reshape(
        d_proto, n_patches, w_proto, h_proto
    )
    closest_patch = patches_with_flat_index[:, min_distance_ix, :, :]
    return closest_patch
