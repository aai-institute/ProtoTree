from copy import copy
from functools import lru_cache
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from proto.img_similarity import ImageProtoSimilarity, img_proto_similarity
from proto.models import TreeSection, ProtoTree, ProtoBase
from proto.node import InternalNode


@torch.no_grad()
def updated_proto_patch_matches(
    base: ProtoBase, original_matches: dict[int, ImageProtoSimilarity], x: torch.Tensor, y: torch.Tensor
) -> dict[int, ImageProtoSimilarity]:
    """
    Produces a map where each key is a node and the corresponding value is information about the patch (out of all
    images in the dataset) that is most similar to node's prototype.

    :param base:
    :param original_matches:
    :param x:
    :param y:
    :return: The map of nodes to best matches.
    """
    updated_matches = copy(original_matches)
    for proto_similarity, label in _patch_match_candidates(base, x, y):
        proto_id = proto_similarity.proto_id
        if proto_id in updated_matches:
            cur_closest = updated_matches[proto_id]
            if (
                proto_similarity.closest_patch_distance
                < cur_closest.closest_patch_distance
            ):
                updated_matches[proto_id] = proto_similarity
        else:
            updated_matches[proto_id] = proto_similarity

    return updated_matches


@torch.no_grad()
def _patch_match_candidates(
    base: ProtoBase, x: torch.Tensor, y: torch.Tensor,
) -> Iterator[Tuple[ImageProtoSimilarity, int]]:
    # TODO: Lots of overlap with Prototree.rationalize, so there's potential for extracting out
    #  commonality. However, we also need to beware of premature abstraction.
    """
    Generator yielding the [node prototype]-[image] similarity (ImageProtoSimilarity) for every (node, image) pair in
    the given tree and dataloader. A generator is used to avoid OOMing on larger datasets and trees.

    :return: Iterator of (similarity, label)
    """
    patches, dists = base.patches(x), base.distances(
        x
    )  # Common subexpression elimination possible, if necessary.

    for x_i, y_i, dists_i, patches_i in zip(x, y, dists, patches):
        for proto_id in range(base.num_prototypes):
            node_distances = dists_i[proto_id, :, :]
            similarity = img_proto_similarity(
                proto_id, x_i, node_distances, patches_i
            )
            yield similarity, y_i
