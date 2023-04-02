import os
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from prototree.node import InternalNode
from prototree.project import ProjectionPatchInfo
from util.data import get_inverse_base_transform


# Adapted from ProtoPNet
# TODO: refactor, use the feature pixels visual field instead of upsampling to some size
@torch.no_grad()
def save_patch_visualizations(
    node_to_patch_info: dict[InternalNode, ProjectionPatchInfo],
    save_path: os.PathLike,
    img_size=(224, 224),
):
    """
    :param node_to_patch_info:
    :param save_path:
    :param img_size: size of images that were used to train the model, i.e. the input to `resize` in the transforms.
        Will be used to create the inverse transform.
    :return:
    """
    save_path = Path(save_path)
    inverse_transform = get_inverse_base_transform(img_size=img_size)

    def latent_to_pixel(latent_img: np.ndarray):
        return cv2.resize(
            latent_img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC
        )

    def save(img: np.ndarray, fname: os.PathLike):
        path = save_path / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            path,
            img,
            vmin=0.0,
            vmax=1.0,
        )

    # TODO: maybe this can be vectorized
    for node, patch_info in node_to_patch_info.items():
        similarity_map = patch_info.get_similarities_latent().cpu().numpy()

        # a single pixel is selected
        # TODO: there is probably a better way to get this mask
        # TODO: What if there's multiple maxima, won't the bounding box be wrong?
        closest_patch_latent_mask = np.uint8(similarity_map == similarity_map.max())
        closest_patch_pixel_mask = latent_to_pixel(closest_patch_latent_mask)
        h_low, h_high, w_low, w_high = covering_rectangle_indices(closest_patch_pixel_mask)

        original_image_unscaled = inverse_transform(patch_info.transformed_image)
        original_image = np.array(original_image_unscaled, dtype=np.float32) / 255

        closest_patch_pixels = original_image[
            h_low:h_high,
            w_low:w_high,
            :,
        ]
        save(closest_patch_pixels, f"{node.index}_closest_patch.png")

        im_with_bbox = get_im_with_bbox(original_image, h_low, h_high, w_low, w_high)
        save(im_with_bbox, f"{node.index}_bounding_box_closest_patch.png")

        pixel_heatmap = latent_to_pixel(similarity_map)
        colored_heatmap = _apply_color_map(pixel_heatmap)
        overlaid_original_img = 0.5 * original_image + 0.2 * colored_heatmap
        save(overlaid_original_img, f"{node.index}_heatmap_original_image.png")


def covering_rectangle_indices(mask: np.ndarray):
    """
    Assuming that mask contains a single connected component with ones, find the indices of the
    smallest rectangle that covers the component.
    :param mask: 2D array of ones and zeros
    :return: indices of the smallest rectangle that covers the component
    """
    nonzero_indices = mask.nonzero()
    lower = np.min(nonzero_indices, axis=1)
    upper = np.max(nonzero_indices, axis=1)
    return lower[0], upper[0] + 1, lower[1], upper[1] + 1


def get_im_with_bbox(
    img: np.ndarray,
    h_low: int,
    h_high: int,
    w_low: int,
    w_high: int,
    color=(0, 255, 255),
):
    """
    :param img: 3D array of shape (H, W, 3). Assumed to be floats, in range [0, 1], and in RGB format.
    :param h_low:
    :param h_high:
    :param w_low:
    :param w_high:
    :param color:
    :return:
    """
    img = np.uint8(255 * img)
    img = cv2.rectangle(
        img,
        pt1=(w_low, h_low),
        pt2=(w_high, h_high),
        color=color,
        thickness=2,
    )
    img = np.float32(img) / 255
    return img


# TODO: maybe optimize? Why is this necessary at all?
def _apply_color_map(arr: Union[torch.Tensor, np.ndarray]):
    """
    Applies cv2 colormap to a 2D array, I suppose for better visibility.
    There might be redundant operations here, I extracted this from the original code.

    :param arr:
    :return:
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    arr = np.uint8(255 * arr)
    arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    arr = np.float32(arr) / 255
    # What's this, why do we need it and the ellipsis? Is this switching the RGB channels?
    arr = arr[..., ::-1]
    return arr
