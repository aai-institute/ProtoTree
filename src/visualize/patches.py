import os
from pathlib import Path

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
    save_dir: os.PathLike,
    img_size=(224, 224),
):
    """
    :param node_to_patch_info:
    :param save_dir:
    :param img_size: size of images that were used to train the model, i.e. the input to `resize` in the transforms.
        Will be used to create the inverse transform.
    :return:
    """
    save_dir = Path(save_dir)
    inverse_transform = get_inverse_base_transform(img_size=img_size)

    def latent_to_pixel(latent_img: np.ndarray):
        return cv2.resize(
            latent_img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC
        )

    def save(img: np.ndarray, fname: str):
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
        similarity_map = patch_info.get_similarities_latent().cpu().numpy()

        # a single pixel is selected
        # TODO: there is probably a better way to get this mask
        closest_patch_latent_mask = np.uint8(similarity_map == similarity_map.max())
        closest_patch_pixel_mask = latent_to_pixel(closest_patch_latent_mask)
        h_low, h_high, w_low, w_high = covering_rectangle_indices(
            closest_patch_pixel_mask
        )

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
        colored_heatmap = _to_rgb_map(pixel_heatmap)
        overlaid_original_img = 0.5 * original_image + 0.2 * colored_heatmap
        save(overlaid_original_img, f"{node.index}_heatmap_original_image.png")


def covering_rectangle_indices(mask: np.ndarray):
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


def get_im_with_bbox(
    img: np.ndarray,
    h_low: int,
    h_high: int,
    w_low: int,
    w_high: int,
    bbox_color=(0, 255, 255),
):
    """
    Takes a 3D float array of shape (H, W, 3), range [0, 1], and RGB format, and superimposes a bounding box.
    """
    img = np.uint8(255 * img)
    img = cv2.rectangle(
        img,
        pt1=(w_low, h_low),
        pt2=(w_high, h_high),
        color=bbox_color,
        thickness=2,
    )
    img = np.float32(img) / 255
    return img


def _to_rgb_map(arr: np.ndarray):
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
