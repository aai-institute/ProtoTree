import torch

from proto.img_similarity import ImageProtoSimilarity
from proto.models import ProtoBase


@torch.no_grad()
def project_prototypes(
    base: ProtoBase, node_to_patch_matches: dict[int, ImageProtoSimilarity]
):
    """
    Replaces each prototype with a given patch.
    Note: This mutates the prototype tensors.
    TODO: We should probably not be mutating the tree (via the prototypes) after training, as this is making the code
     less flexible and harder to reason about.
    """
    for proto_id, patch_info in node_to_patch_matches.items():
        base.proto_layer.protos.data[proto_id] = patch_info.closest_patch.data
