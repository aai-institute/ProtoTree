import logging
import os
from typing import Iterator, Callable

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from src.core.models import ProtoTree
from src.util.data import save_img
from src.util.image import get_inverse_arr_transform, get_latent_to_pixel
from src.visualize.create.patches import (
    _bbox_indices,
    _to_rgb_heatmap,
    _superimpose_bboxs,
    _bbox_color,
    Opacity,
    Bbox,
)

log = logging.getLogger(__name__)


# @torch.no_grad()
# def save_multi_patch_visualizations(
#     explanations: Iterator[tuple[ProtoTree.LeafRationalization, str, tuple]],
#     explanations_dir: os.PathLike,
#     img_size=(224, 224),
# ):
#     """
#     Saves visualizations of each explanation as a DOT file and png. In these visualizations, every patch in an
#     explanation is applied to a single copy of the image.
#     TODO: Note that this currently relies on the patch visualizations being run first. We should probably change this,
#      or change the API to enforce it.
#     """
#     multi_patches_dir = explanations_dir / "multi_patch"
#     multi_patches_dir.mkdir(parents=True, exist_ok=True)
#     inv_transform = get_inverse_arr_transform(img_size)
#     latent_to_pixel = get_latent_to_pixel(img_size)

#     log.info(
#         f"Saving multiple patch visualizations of the explanations to {multi_patches_dir}."
#     )
#     tqdm_explanations = tqdm(
#         explanations,
#         desc="Saving multiple patch visualizations of the explanations",
#         ncols=0,
#     )
#     for explanation_counter, (leaf_explanation, true_class, class_names) in enumerate(
#         tqdm_explanations
#     ):
#         multi_patch_dir = multi_patches_dir / f"img_{explanation_counter}"
#         _save_multi_patch_vis(
#             leaf_explanation,
#             inv_transform,
#             latent_to_pixel,
#             multi_patch_dir,
#         )
@torch.no_grad()
def save_multi_patch_visualizations(
    explanations: Iterator[tuple[ProtoTree.LeafRationalization, str, tuple]],
    explanations_dir: os.PathLike,
    local_scores: pd.DataFrame = None,
    img_size=(224, 224),
):
    """
    Saves visualizations of each explanation as a DOT file and png. In these visualizations, every patch in an
    explanation is applied to a single copy of the image.
    TODO: Note that this currently relies on the patch visualizations being run first. We should probably change this,
     or change the API to enforce it.
    """
    multi_patches_dir = explanations_dir / "multi_patch"
    multi_patches_dir.mkdir(parents=True, exist_ok=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    log.info(
        f"Saving multiple patch visualizations of the explanations to {multi_patches_dir}."
    )
    tqdm_explanations = tqdm(
        explanations,
        desc="Saving multiple patch visualizations of the explanations",
        ncols=0,
    )
    for explanation_counter, sample in enumerate(
        tqdm_explanations
    ):
        leaf_explanation, true_class, class_names = sample[0], sample[1], sample[2]
        if local_scores is not None:
            img_path = sample[3]
            multi_patch_dir = multi_patches_dir / os.path.basename(img_path).split(".")[0]
        else:
            multi_patch_dir = multi_patches_dir / f"img_{explanation_counter}"
        
        _save_multi_patch_vis(
            leaf_explanation,
            inv_transform,
            latent_to_pixel,
            multi_patch_dir,
            local_scores
        )

def _save_multi_patch_vis(
    leaf_rationalization: ProtoTree.LeafRationalization,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    multi_patch_dir: os.PathLike,
    local_score: pd.DataFrame = None
):
    """
    Saves the original image and copies of it with {an average heatmap, all bounding boxes from patches, bounding boxes
    from patches that were similar enough to be considered present}.
    """
    ancestor_sims = leaf_rationalization.ancestor_sims
    transformed_orig = ancestor_sims[0].similarity.transformed_image
    im_original = inv_transform(transformed_orig)

    # TODO: Seems a bit redundant that we're extracting the similarities and then max similarities separately.
    all_patch_similarities = [
        sim.similarity.all_patch_similarities.cpu().numpy() for sim in ancestor_sims
    ]
    highest_similarities = [
        ancestor_sim.similarity.highest_patch_similarity
        for ancestor_sim in ancestor_sims
    ]

    im_with_bboxs, im_with_present_bboxs = _bboxs_overlaid(
        all_patch_similarities,
        highest_similarities,
        leaf_rationalization.proto_presents(),
        latent_to_pixel,
        im_original,
    )
    im_with_heatmap = _avg_similarity_heatmap(
        all_patch_similarities, latent_to_pixel, im_original
    )

    save_img(im_original, multi_patch_dir / "original.png")
    save_img(im_with_bboxs, multi_patch_dir / "im_with_bboxs.png")
    save_img(im_with_present_bboxs, multi_patch_dir / "im_with_present_bboxs.png")
    save_img(im_with_heatmap, multi_patch_dir / "im_with_heatmap.png")


def _bboxs_overlaid(
    all_patch_similarities: list[np.ndarray],
    highest_similarities: list[float],
    proto_presents: list[bool],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    im_original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produces copies of the original image overlaid with {all bounding boxes from patches, bounding boxes from patches
    that were similar enough to be considered present}. Present bounding boxes are green, absent boxes are red.
    """
    bboxs_inds = [
        _bbox_indices(sims, latent_to_pixel) for sims in all_patch_similarities
    ]
    bboxs_color = [_bbox_color(sim) for sim in highest_similarities]
    bboxs_opacities = [Opacity(sim) for sim in highest_similarities]
    bboxs = [
        Bbox(inds, color, opacity)
        for inds, color, opacity in zip(bboxs_inds, bboxs_color, bboxs_opacities)
    ]
    present_bboxs = [bbox for bbox, present in zip(bboxs, proto_presents) if present]

    im_with_bboxs = _superimpose_bboxs(im_original, bboxs)
    im_with_present_bboxs = _superimpose_bboxs(im_original, present_bboxs)

    return im_with_bboxs, im_with_present_bboxs


def _avg_similarity_heatmap(
    all_patch_similarities: list[np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    im_original: np.ndarray,
) -> np.ndarray:
    """
    Returns the original image with an average similarity heatmap overlaid.
    """
    pixel_heatmaps = [latent_to_pixel(sim) for sim in all_patch_similarities]
    avg_pixel_heatmap = np.mean(pixel_heatmaps, axis=0)
    avg_pixel_rgb_heatmap = _to_rgb_heatmap(avg_pixel_heatmap)
    return 0.5 * im_original + 0.2 * avg_pixel_rgb_heatmap
