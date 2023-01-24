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


# adapted from protopnet
# TODO: refactor, use the feature pixels visual field instead of upsampling to some size
@torch.no_grad()
def save_prototype_visualizations(
    node_to_patch_info: dict[InternalNode, ProjectionPatchInfo],
    save_dir: os.PathLike,
):
    save_dir = Path(save_dir)
    inverse_transform = get_inverse_base_transform()

    def upsample(img: np.ndarray, img_size: tuple[int, int]):
        return cv2.resize(
            img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC
        )

    def save(img: np.ndarray, fname: os.PathLike):
        path = save_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            path,
            img,
            vmin=0.0,
            vmax=1.0,
        )

    # TODO: maybe this can be vectorized
    for node, patch_info in node_to_patch_info.items():
        x = inverse_transform(patch_info.image)
        x = np.array(x)
        img_size = x.shape[:2]

        x = np.float32(x) / 255
        # TODO: are there grayscale images?
        if x.ndim == 2:  # convert grayscale to RGB
            x = np.stack((x,) * 3, axis=-1)

        similarity_map = torch.exp(-patch_info.all_patch_distances).cpu().numpy()

        heatmap_in_latent = _apply_color_map(similarity_map)
        save(heatmap_in_latent, f"{node.index}_heatmap_latent.png")

        heatmap_in_pixels = upsample(similarity_map, img_size)
        heatmap_in_pixels = _apply_color_map(heatmap_in_pixels)
        overlayed_original_img = 0.5 * x + 0.2 * heatmap_in_pixels
        save(overlayed_original_img, f"{node.index}_heatmap_original_image.png")

        # a single pixel is selected
        closest_patch_mask_pixels = np.uint8(similarity_map == similarity_map.max())
        closest_patch_mask_pixels = upsample(closest_patch_mask_pixels, img_size)
        # save(closest_patch_mask_pixels, f"{node.index}_closest_patch_mask.png")

        # TODO: ugly as hell, refactor. We use a mask here, threshold has no meaning...
        #
        h_low, h_high, w_low, w_high = covering_rectangle_indices(
            closest_patch_mask_pixels
        )
        # closest_patch_in_pixels = x[
        #     h_low:h_high,
        #     w_low:w_high,
        #     :,
        # ]
        # save(closest_patch_in_pixels, f"{node.index}_closest_patch.png")

        im_with_bbox = get_im_with_bbox(x, h_low, h_high, w_low, w_high)
        save(im_with_bbox, f"{node.index}_bounding_box_closes_patch.png")


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


# copied from protopnet
# TODO: replace by covering_rectangle_indices
def find_high_activation_crop(mask: np.ndarray, threshold: float):
    threshold = 1.0 - threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


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
    # img = img[..., ::-1]
    img = np.float32(img) / 255
    return img


# copied from protopnet
# TODO: no input validation, saving and creating bbox coupled. Refactor or make private.
# TODO: use same bbox conventions as elsewhere in code
def imsave_with_bbox(
    path: os.PathLike,
    img: np.ndarray,
    h_low: int,
    h_high: int,
    w_low: int,
    w_high: int,
    color=(0, 255, 255),
):
    """

    :param path:
    :param img: assumed to be in range [0, 1]
    :param h_low:
    :param h_high:
    :param w_low:
    :param w_high:
    :param color:
    :return:
    """
    img = get_im_with_bbox(img, h_low, h_high, w_low, w_high, color)
    plt.imsave(path, img)
