import logging
import os
from typing import Iterator, Callable

import numpy as np
import pydot
import torch
from tqdm import tqdm

from prototree.models import LeafRationalization
from util.data import save_img
from util.image import get_inverse_arr_transform, get_latent_to_pixel
from visualize.create.patches import _bbox_indices, GREEN_RGB, RED_RGB, _to_rgb_heatmap, _superimpose_bboxs

log = logging.getLogger(__name__)


@torch.no_grad()
def save_multi_bbox_visualizations(
    explanations: Iterator[tuple[LeafRationalization, str, tuple]],
    patches_dir: os.PathLike,
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    """
    Saves visualizations of each explanation as a DOT file and png.
    TODO: Note that this currently relies on the patch visualizations being run first. We should probably change this,
     or change the API to enforce it.
    """
    multi_bboxs_dir = explanations_dir / "multi_bbox"
    multi_bboxs_dir.mkdir(parents=True, exist_ok=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    log.info(
        f"Saving multiple bounding box visualizations of the explanations to {multi_bboxs_dir}."
    )
    tqdm_explanations = tqdm(
        explanations,
        desc="Saving multiple bounding box visualizations of the explanations",
        ncols=0,
    )
    for explanation_counter, (leaf_explanation, true_class, class_names) in enumerate(
        tqdm_explanations
    ):
        multi_bbox_dir = multi_bboxs_dir / f"img_{explanation_counter}"


def _multi_bbox_graph(
    leaf_rationalization: LeafRationalization,
    true_class: str,
    class_names: tuple,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    patches_dir: os.PathLike,
    multi_bbox_dir: os.PathLike,
) -> pydot.Dot:
    bboxs, pixel_heatmaps = [], []
    for ancestor_similarity, proto_present in zip(
        leaf_rationalization.ancestor_similarities,
        leaf_rationalization.proto_presents,
    ):
        patch_similarities = ancestor_similarity.all_patch_similarities.cpu().numpy()

        bbox_inds = _bbox_indices(patch_similarities, latent_to_pixel)
        bbox_color = GREEN_RGB if proto_present else RED_RGB
        bboxs.append((bbox_inds, bbox_color))

        pixel_heatmaps.append(latent_to_pixel(patch_similarities))

    avg_pixel_heatmap = np.mean(pixel_heatmaps, axis=0)
    avg_pixel_rgb_heatmap = _to_rgb_heatmap(avg_pixel_heatmap)

    im_original = inv_transform(leaf_rationalization.ancestor_similarities[0].transformed_image)
    im_with_bboxs = _superimpose_bboxs(im_original, bboxs)
    im_with_heatmap = 0.5 * im_original + 0.2 * avg_pixel_rgb_heatmap

    save_img(im_with_bboxs, multi_bbox_dir / "im_with_bboxs.png")
    save_img(im_with_heatmap, multi_bbox_dir / "im_with_heatmap.png")
