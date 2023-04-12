import logging
import os
from typing import Callable, Iterable

import cv2
import numpy as np
import torch

from prototree.node import InternalNode
from prototree.img_similarity import ImageProtoSimilarity
from util.data import save_img
from util.image import get_latent_to_pixel, get_inverse_arr_transform

log = logging.getLogger(__name__)

# TODO: Should we make dataclasses for these?
BboxInds = tuple[int, int, int, int]
ColorRgb = tuple[int, int, int]
Bbox = tuple[BboxInds, ColorRgb]

RED_RGB: ColorRgb = (255, 0, 0)
YELLOW_RGB: ColorRgb = (0, 255, 255)


@torch.no_grad()
def save_patch_visualizations(
    node_to_patch_matches: dict[InternalNode, ImageProtoSimilarity],
    save_dir: os.PathLike,
    img_size=(224, 224),
):
    # Adapted from ProtoPNet
    """
    :param node_to_patch_matches:
    :param save_dir:
    :param img_size: size of images that were used to train the model, i.e. the input to `resize` in the transforms.
        Will be used to create the inverse transform.
    :return:
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    log.info(f"Saving prototype patch visualizations to {save_dir}.")
    for node, image_proto_similarity in node_to_patch_matches.items():
        (
            im_closest_patch,
            im_original,
            im_with_bbox,
            im_with_heatmap,
        ) = closest_patch_imgs(image_proto_similarity, inv_transform, latent_to_pixel)

        # TODO: These filenames should come from config (same for the other py files).
        save_img(im_closest_patch, save_dir / f"{node.index}_closest_patch.png")
        save_img(
            im_with_bbox, save_dir / f"{node.index}_bounding_box_closest_patch.png"
        )
        save_img(im_with_heatmap, save_dir / f"{node.index}_heatmap_original_image.png")


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
    h_low, h_high, w_low, w_high = bbox_inds
    im_closest_patch = im_original[
        h_low:h_high,
        w_low:w_high,
        :,
    ]

    im_with_bbox = _superimpose_bboxs(im_original, [(bbox_inds, YELLOW_RGB)])

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
    return lower[0], upper[0] + 1, lower[1], upper[1] + 1


def _superimpose_bboxs(
    img: np.ndarray,
    bboxs: Iterable[Bbox]
) -> np.ndarray:
    """
    Takes a 3D float array of shape (H, W, 3), range [0, 1], and RGB format, and superimposes the given bounding boxes
    in the order given.
    """
    img = np.uint8(255 * img)
    for bbox_inds, bbox_color in bboxs:
        h_low, h_high, w_low, w_high = bbox_inds
        img = cv2.rectangle(
            img,
            pt1=(w_low, h_low),
            pt2=(w_high, h_high),
            color=bbox_color,
            thickness=2,
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
