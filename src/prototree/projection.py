import torch

from prototree.img_similarity import ImageProtoSimilarity
from prototree.models import ProtoTree
from prototree.node import Node


@torch.no_grad()
def replace_prototypes_with_patches(
    tree: ProtoTree, node_to_patch_matches: dict[Node, ImageProtoSimilarity]
):
    """
    Replaces each prototype with a given patch.
    Note: This mutates the prototype tensors.
    TODO: We should probably not be mutating the tree (via the prototypes), as this is making the code less flexible and
     harder to reason about.
    """
    for internal_node, patch_info in node_to_patch_matches.items():
        node_proto_idx = tree.node_to_proto_idx[internal_node]
        tree.prototype_layer.prototype_tensors.data[
            node_proto_idx
        ] = patch_info.closest_patch.data
