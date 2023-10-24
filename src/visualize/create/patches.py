import logging
import os
from dataclasses import dataclass, astuple
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import json

from src.core.img_similarity import ImageProtoSimilarity
from src.util.data import save_img
from src.util.image import get_latent_to_pixel, get_inverse_arr_transform

log = logging.getLogger(__name__)


@dataclass
class BboxInds:
    h_low: int
    h_high: int
    w_low: int
    w_high: int


@dataclass
class ColorRgb:
    red: int
    green: int
    blue: int


@dataclass
class Opacity:
    alpha: float


@dataclass
class Bbox:
    inds: BboxInds
    color: ColorRgb
    opacity: Opacity


YELLOW_RGB: ColorRgb = ColorRgb(255, 255, 0)


@torch.no_grad()
def save_patch_visualizations(
    proto_patch_matches: dict[int, ImageProtoSimilarity],
    save_dir: os.PathLike,
    img_size=(224, 224),
    save_as_json= True
):
    # Adapted from ProtoPNet
    """
    :param proto_patch_matches:
    :param save_dir:
    :param img_size: size of images that were used to train the model, i.e. the input to `resize` in the transforms.
        Will be used to create the inverse transform.
    :return:
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    log.info(f"Saving prototype patch visualizations to {save_dir}.")
    if save_as_json:
        prototypes_info = dict()
        for proto_id, image_proto_similarity in proto_patch_matches.items():
            patch_similarities = image_proto_similarity.all_patch_similarities.cpu().numpy()
            bbox_inds = _bbox_indices(patch_similarities, latent_to_pixel)
            
            prototypes_info[proto_id] = dict(patch_similarities=patch_similarities.tolist(),
                                             bbox = list(map(int, [bbox_inds.w_low, 
                                                                   bbox_inds.h_low, 
                                                                   bbox_inds.w_high, 
                                                                   bbox_inds.h_high]
                                             )),
                                             path=image_proto_similarity.path)
            
            with open(save_dir / "proto_info.json", "w") as f:
                json.dump(prototypes_info, f)

    else:
        
        for proto_id, image_proto_similarity in proto_patch_matches.items():
            (
                im_closest_patch,
                im_original,
                im_with_bbox,
                im_with_heatmap,
            ) = closest_patch_imgs(image_proto_similarity, inv_transform, latent_to_pixel)

            # TODO: These filenames should come from config (same for the other py files).
            save_img(im_closest_patch, save_dir / f"{proto_id}_closest_patch.png")
            save_img(im_with_bbox, save_dir / f"{proto_id}_bounding_box_closest_patch.png")
            save_img(im_with_heatmap, save_dir / f"{proto_id}_heatmap_original_image.png")


@torch.no_grad()
def closest_patch_imgs(
    image_proto_similarity: ImageProtoSimilarity,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # TODO: Use the patch receptive fields instead of upsampling.
    """
    Gets the pixels for images illustrating the closest patch from an ImageProtoSimilarity.
    :return: Pixels for: (closest patch, original image, original image with bounding box, original image with heatmap)
    """
    patch_similarities = image_proto_similarity.all_patch_similarities.cpu().numpy()

    im_original = inv_transform(image_proto_similarity.transformed_image)

    bbox_inds = _bbox_indices(patch_similarities, latent_to_pixel)
    im_closest_patch = im_original[
        bbox_inds.h_low : bbox_inds.h_high,
        bbox_inds.w_low : bbox_inds.w_high,
        :,
    ]

    im_with_bbox = _superimpose_bboxs(
        im_original, [Bbox(bbox_inds, YELLOW_RGB, Opacity(1.0))]
    )

    pixel_heatmap = latent_to_pixel(patch_similarities)
    colored_heatmap = _to_rgb_heatmap(pixel_heatmap)
    im_with_heatmap = 0.5 * im_original + 0.2 * colored_heatmap

    return im_closest_patch, im_original, im_with_bbox, im_with_heatmap


def _bbox_indices(
    patch_similarities: np.ndarray,
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
) -> BboxInds:
    # A single pixel is selected.
    # Max because this similarity measure is higher for more similar patches.
    closest_patch_latent_mask = np.uint8(patch_similarities == patch_similarities.max())
    closest_patch_pixel_mask = latent_to_pixel(closest_patch_latent_mask)
    return _covering_bbox_inds(closest_patch_pixel_mask)


def _covering_bbox_inds(mask: np.ndarray) -> BboxInds:
    """
    Assuming that mask contains a single connected component with ones, find the indices of the
    smallest rectangle that covers the component.
    TODO: Handle the case of multiple maxima? (not urgent since it's very unlikely on real data)
    :param mask: 2D array of ones and zeros
    :return: indices of the smallest rectangle that covers the component
    """
    nonzero_indices = mask.nonzero()
    lower = np.min(nonzero_indices, axis=1)
    upper = np.max(nonzero_indices, axis=1)
    return BboxInds(lower[0], upper[0] + 1, lower[1], upper[1] + 1)


def _superimpose_bboxs(img: np.ndarray, bboxs: Iterable[Bbox]) -> np.ndarray:
    """
    Takes a 3D float array of shape (H, W, 3), range [0, 1], and RGB format, and superimposes the given bounding boxes
    in the order given.
    """
    img = np.uint8(255 * img)
    for bbox in bboxs:
        overlay = img.copy()
        overlay = cv2.rectangle(
            overlay,
            pt1=(bbox.inds.w_low, bbox.inds.h_low),
            pt2=(bbox.inds.w_high, bbox.inds.h_high),
            color=astuple(bbox.color),
            thickness=2,
        )
        img = cv2.addWeighted(
            overlay, bbox.opacity.alpha, img, 1.0 - bbox.opacity.alpha, 0.0
        )
    img = np.float32(img) / 255
    return img


def _to_rgb_heatmap(arr: np.ndarray) -> np.ndarray:
    """
    Turns a single-channel heatmap into an RGB heatmap.
    """
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    arr = np.uint8(255 * arr)
    arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    arr = np.float32(arr) / 255
    arr = arr[
        :, :, ::-1
    ]  # Reverse channels, so red covers the most similar patches (the highest similarity values).
    return arr


def _bbox_color(similarity: float) -> ColorRgb:
    """
    Takes a similarity float between 0 and 1 (inclusive) and maps it to colors ranging from red for 0, to yellow
    for 0.5, to green for 1.
    """
    assert 0.0 <= similarity <= 1.0

    if similarity <= 0.5:
        interpolator = similarity * 2.0
        green_component = int(255 * interpolator)
        return ColorRgb(255, green_component, 0)

    interpolator = (1.0 - similarity) * 2.0
    red_component = int(interpolator * 255)
    return ColorRgb(red_component, 255, 0)
