import os
from collections.abc import Callable
from pathlib import Path
from typing import Iterator

import numpy as np
import pydot
import torch
from PIL.Image import Image

from prototree.models import LeafRationalization
from util.image import get_inverse_base_transform, get_latent_to_pixel


@torch.no_grad()
def save_explanation_visualizations(
    explanations: Iterator[tuple[LeafRationalization, int, tuple]],
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    explanations_dir = Path(explanations_dir)
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

    for ancestor_similarity in leaf_rationalization.ancestor_similarities:
        ancestor_similarity.transformed_image
