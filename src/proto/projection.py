import torch

from proto.img_similarity import ImageProtoSimilarity
from proto.models import TreeSection
from proto.node import Node


@torch.no_grad()
def project_prototypes(
    tree: TreeSection, node_to_patch_matches: dict[Node, ImageProtoSimilarity]
):
    """
    Replaces each prototype with a given patch.
    Note: This mutates the prototype tensors.
    TODO: We should probably not be mutating the tree (via the prototypes) after training, as this is making the code
     less flexible and harder to reason about.
    """
    for internal_node, patch_info in node_to_patch_matches.items():
        node_proto_idx = tree.node_to_proto_idx[internal_node]
        tree.prototype_layer.protos.data[
            node_proto_idx
        ] = patch_info.closest_patch.data
