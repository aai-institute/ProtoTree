from typing import Optional, Union, Literal, Iterator, Tuple

import torch
from torch import nn as nn

from src.core.img_similarity import ImageProtoSimilarity, img_proto_similarity
from src.util.l2conv import L2Conv2D
from src.util.net import default_add_on_layers


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
        """
        Get Tuple[patches for a given input tensor, minimal distances between the prototypes and the input]
        This is just an optimized version of the patches and distances methods so that we don't have to extract the
        features from the backbone twice.

        Outputs have shapes:
            (batch_size, d, n_patches_w, n_patches_h, w_proto, h_proto)
            (batch_size, num_prototypes, n_patches_w, n_patches_h)
        """
        features = self.extract_features(x)
        return self._features_to_patches(features), self.proto_layer(features)

    def _features_to_patches(self, features: torch.Tensor) -> torch.Tensor:
        w_proto, h_proto = self.prototype_shape[:2]
        return features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Transformed images (batch_size, num_channels, W, H)

        Returns:
            Minimum distances between each image & prototype pair (batch_size, num_prototypes)
        """
        dists = self.distances(x)
        return torch.amin(dists, dim=(2, 3))

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
    def project_prototypes(self, proto_patch_patches: dict[int, ImageProtoSimilarity]):
        # TODO: We should probably not be mutating the model (via the prototypes) after training, as this is making the
        #  code less flexible and harder to reason about.
        
        
        ### On the idea of changing model part from the paper
        """
        Replaces each prototype with a given patch.
        Note: This mutates the prototype tensors, and hence modifies any model using them.

        The intended usage of this is that after collecting a map of proto_idx to most similar image patches elsewhere
        in the code (presumably by looping through the data with update_proto_patch_matches), the caller can call this
        method to update the prototypes to be the most similar patches from actual images.

        :param proto_patch_patches: The current map of proto_idx to data on the most similar image patch. Note that it
         will be mutated by this method.
        """
        for proto_id, patch_info in proto_patch_patches.items():
            self.proto_layer.protos.data[proto_id] = patch_info.closest_patch.data

    @torch.no_grad()
    def update_proto_patch_matches(
        self,
        proto_patch_patches: dict[int, ImageProtoSimilarity],
        x: torch.Tensor,
        path: torch.Tensor = None
    ):
        # TODO: This is currently incredibly slow, particularly on GPUs, because of the large number of small,
        #  non-vectorized operations. This can probably be refactored to be much faster.
        #  Another small thing is that this function isn't pure, it mutates the dictionary since even shallow copying
        #  can be a bit slow; is there a more functional way of doing this?
        
        ###  problem of mutable dictionary
        """
        Takes a map where each key is a proto_idx and the corresponding value is information about the patch that is
        most similar to the prototype, and then updates that map based on any new best matches (most similar patches)
        found in the new data x. The intended use for this is that the caller will start with an empty dictionary, and
        then repeatedly pass the dictionary into this method while looping through batches of data, in order to modify
        this dictionary to keep it populated with the best matching patches from images for each prototype.

        :param proto_patch_patches: The current map of proto_idx to data on the most similar image patch. Note that it
         will be mutated by this method.
        :param x: A batch of images.
        """
        for proto_similarity in self._patch_match_candidates(x, path):
            proto_id = proto_similarity.proto_id
            if proto_id in proto_patch_patches:
                cur_closest = proto_patch_patches[proto_id]
                if (
                    proto_similarity.closest_patch_distance
                    < cur_closest.closest_patch_distance
                ):
                    proto_patch_patches[proto_id] = proto_similarity
            else:
                proto_patch_patches[proto_id] = proto_similarity

    @torch.no_grad()
    def _patch_match_candidates(
        self,
        x: torch.Tensor,
        path: torch.Tensor = None
    ) -> Iterator[ImageProtoSimilarity]:
        # TODO: Lots of overlap with Prototree.rationalize, so there's potential for extracting out
        #  commonality. However, we also need to beware of premature abstraction.
        """
        Generator yielding the [prototype]-[image] similarity (ImageProtoSimilarity) for every (proto, image) pair
        in the given tree and dataloader. A generator is used to avoid OOMing on larger datasets and trees.

        :return: Iterator of (similarity, label)
        """
        patches, dists = self.patches_and_dists(x)

        for i, (x_i, dists_i, patches_i) in enumerate(zip(x, dists, patches)):
            for proto_id in range(self.num_prototypes):
                patch_distances = dists_i[proto_id, :, :]
                if path:
                    yield img_proto_similarity(proto_id, x_i, patch_distances, patches_i, path[i])
                else:
                    yield img_proto_similarity(proto_id, x_i, patch_distances, patches_i)
