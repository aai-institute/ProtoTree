from functools import lru_cache
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.img_similarity import ImageProtoSimilarity, img_proto_similarity
from prototree.models import ProtoTree
from prototree.node import InternalNode


@torch.no_grad()
def node_patch_matches(
    tree: ProtoTree,
    loader: DataLoader,
    constrain_on_classes=True,
) -> dict[InternalNode, ImageProtoSimilarity]:
    # TODO: Generalize to ProtoBase
    """
    Produces a map where each key is a node and the corresponding value is information about the patch (out of all
    images in the dataset) that is most similar to node's prototype.

    :param tree:
    :param loader: The dataset.
    :param constrain_on_classes: If True, only consider patches from classes that are contained in
        the prototype's leaves' predictions.
    :return: The map of nodes to best matches.
    """

    @lru_cache(maxsize=10000)
    def get_leaf_labels(internal_node: InternalNode):
        # TODO: Should this be a method on the node? If this weren't an inner function, we'd need to beware of caching
        #  incorrect results when the leaf logits change.
        return {leaf.predicted_label() for leaf in internal_node.leaves}

    # TODO: (Minor) Is there a more functional way of doing this?
    node_to_patch_matches: dict[InternalNode, ImageProtoSimilarity] = {}
    for proto_similarity, label in _patch_match_candidates(tree, loader):
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


@torch.no_grad()
def _patch_match_candidates(
    tree: ProtoTree, loader: DataLoader
) -> Iterator[Tuple[ImageProtoSimilarity, int]]:
    # TODO: Lots of overlap with Prototree.rationalize, but we need to beware of premature abstraction.
    """
    Generator yielding the [node prototype]-[image] similarity (ImageProtoSimilarity) for every (node, image) pair in
    the given tree and dataloader. A generator is used to avoid OOMing on larger datasets and trees.

    :return: Iterator of (similarity, label)
    """
    for x, y in tqdm(loader, desc="Data loader", ncols=0):
        x, y = x.to(tree.device), y.to(tree.device)
        patches, dists = tree.patches(x), tree.distances(
            x
        )  # Common subexpression elimination possible, if necessary.

        for x_i, y_i, dists_i, patches_i in zip(x, y, dists, patches):
            for internal_node in tree.internal_nodes:
                node_proto_idx = tree.node_to_proto_idx[internal_node]

                node_distances = dists_i[node_proto_idx, :, :]
                similarity = img_proto_similarity(
                    internal_node, x_i, node_distances, patches_i
                )
                yield similarity, y_i
