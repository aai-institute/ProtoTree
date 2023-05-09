from typing import Optional, Union, Literal, Iterator, Tuple

import torch
from torch import nn as nn

from prototree.img_similarity import ImageProtoSimilarity, img_proto_similarity
from util.l2conv import L2Conv2D
from util.net import default_add_on_layers


class ProtoBase(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        prototype_shape: tuple[int, int, int],
        backbone: torch.nn.Module,
        add_on_layers: Optional[Union[nn.Module, Literal["default"]]] = "default",
    ):
        """
        :param prototype_shape: shape of the prototypes. (channels, height, width)
        :param backbone: usually a pretrained network that extracts features from the input images
        :param add_on_layers: used to connect the feature net with the prototypes.
        """
        super().__init__()

        # TODO: The parameters in this hardcoded initialization seem to (very, very) roughly correspond to a random
        #  average-looking latent patch (which is a good start point for the prototypes), but it would be nice if we had
        #  a more principled way of choosing the initialization.
        # NOTE: The paper means std=0.1 when it says N(0.5, 0.1), not var=0.1.
        self.proto_layer = L2Conv2D(
            num_prototypes, *prototype_shape, initial_mean=0.5, initial_std=0.1
        )

        if isinstance(add_on_layers, nn.Module):
            self.add_on = add_on_layers
        elif add_on_layers is None:
            self.add_on = nn.Identity()
        elif add_on_layers == "default":
            self.add_on = default_add_on_layers(backbone, prototype_shape[0])
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feature net and add_on layers to the input. The output
        has the shape (batch_size, num_channels, height, width), where num_channels is the
        number of channels of the prototypes.
        """
        x = self.backbone(x)
        x = self.add_on(x)
        return x

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the patches for a given input tensor. This is the same as extract_features, except the output is unfolded to
        be (batch_size, d, n_patches_w, n_patches_h, w_proto, h_proto).
        """
        features = self.extract_features(x)
        return self._features_to_patches(features)

    def distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the minimal distances between the prototypes and the input.
        The output has the shape (batch_size, num_prototypes, n_patches_w, n_patches_h)
        """
        x = self.extract_features(x)
        return self.proto_layer(x)

    def patches_and_dists(self, x: torch.Tensor):
        features = self.extract_features(x)
        return self._features_to_patches(features), self.proto_layer(features)

    def _features_to_patches(self, features: torch.Tensor) -> torch.Tensor:
        w_proto, h_proto = self.prototype_shape[:2]
        return features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.distances(x)
        return torch.amin(x, dim=(2, 3))

    @property
    def num_prototypes(self):
        return self.proto_layer.num_protos

    @property
    def prototype_channels(self):
        return self.proto_layer.input_channels

    @property
    def prototype_shape(self):
        return self.proto_layer.proto_shape

    @torch.no_grad()
    def project_prototypes(
        self, node_to_patch_matches: dict[int, ImageProtoSimilarity]
    ):
        """
        Replaces each prototype with a given patch.
        Note: This mutates the prototype tensors.
        TODO: We should probably not be mutating the tree (via the prototypes) after training, as this is making the
         code less flexible and harder to reason about.
        """
        for proto_id, patch_info in node_to_patch_matches.items():
            self.proto_layer.protos.data[proto_id] = patch_info.closest_patch.data

    @torch.no_grad()
    def update_proto_patch_matches(
        self,
        updated_matches: dict[int, ImageProtoSimilarity],
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        # TODO: This is currently incredibly slow, particularly on GPUs, because of the large number of small,
        #  non-vectorized operations. This can probably be refactored to be much faster.
        """
        Produces a map where each key is a node and the corresponding value is information about the patch (out of all
        images in the dataset) that is most similar to node's prototype.

        :param updated_matches:
        :param x:
        :param y:
        :return: The map of nodes to best matches.
        """
        for proto_similarity, label in self._patch_match_candidates(x, y):
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

    @torch.no_grad()
    def _patch_match_candidates(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Iterator[Tuple[ImageProtoSimilarity, int]]:
        # TODO: Lots of overlap with Prototree.rationalize, so there's potential for extracting out
        #  commonality. However, we also need to beware of premature abstraction.
        """
        Generator yielding the [node prototype]-[image] similarity (ImageProtoSimilarity) for every (node, image) pair
        in the given tree and dataloader. A generator is used to avoid OOMing on larger datasets and trees.

        :return: Iterator of (similarity, label)
        """
        patches, dists = self.patches_and_dists(x)

        for x_i, y_i, dists_i, patches_i in zip(x, y, dists, patches):
            for proto_id in range(self.num_prototypes):
                node_distances = dists_i[proto_id, :, :]
                similarity = img_proto_similarity(proto_id, x_i, node_distances, patches_i)
                yield similarity, y_i
