import os
from collections.abc import Callable
from pathlib import Path
from typing import Iterator

import numpy as np
import pydot
import torch
from PIL.Image import Image

from prototree.models import LeafRationalization
from prototree.node import Node
from util.data import save_img
from util.image import get_inverse_base_transform, get_latent_to_pixel
from visualize.create.patches import closest_patch_imgs


@torch.no_grad()
def save_explanation_visualizations(
    explanations: Iterator[tuple[LeafRationalization, int, tuple]],
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    inverse_transform = get_inverse_base_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    for explanation_counter, (leaf_explanation, true_label, class_names) in enumerate(
        explanations
    ):
        explanation_dir = explanations_dir / str(explanation_counter)
        pydot_graph = _explanation_pydot(
            leaf_explanation,
            true_label,
            class_names,
            inverse_transform,
            latent_to_pixel,
            explanation_dir,
        )


def _explanation_pydot(
    leaf_rationalization: LeafRationalization,
    true_label: int,
    class_names: tuple,
    inverse_transform: Callable[[torch.Tensor], Image],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    explanation_dir: os.PathLike,
) -> pydot.Dot:
    dag = pydot.Dot(
        "Decision flow for explanation of 1 image.",
        graph_type="digraph",
    )

    for ancestor_similarity, went_right in zip(
        leaf_rationalization.ancestor_similarities,
        leaf_rationalization.ancestors_went_right,
    ):
        (
            im_closest_patch,
            im_original,
            im_with_bbox,
            im_with_heatmap,
        ) = closest_patch_imgs(ancestor_similarity, inverse_transform, latent_to_pixel)

        save_img(
            im_with_bbox,
            explanation_dir
            / f"level_{ancestor_similarity.internal_node.depth}_bounding_box_patch.png",
        )
        # ancestor_similarity.all_patch_similarities()
