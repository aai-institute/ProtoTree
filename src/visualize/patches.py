import os
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image

from prototree.node import InternalNode
from prototree.img_similarity import ImageProtoSimilarity
from util.data import get_inverse_base_transform


# Adapted from ProtoPNet
@torch.no_grad()
def save_patch_visualizations(
    node_to_patch_matches: dict[InternalNode, ImageProtoSimilarity],
    save_dir: os.PathLike,
    img_size=(224, 224),
):
    """
    :param node_to_patch_matches:
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

    for node, image_proto_similarity in node_to_patch_matches.items():
        (
            im_closest_patch,
            im_original,
            im_with_bbox,
            im_with_heatmap,
        ) = get_closest_patch_imgs(
            image_proto_similarity, inverse_transform, latent_to_pixel
        )
        save(im_closest_patch, f"{node.index}_closest_patch.png")
        save(im_with_bbox, f"{node.index}_bounding_box_closest_patch.png")
        save(im_with_heatmap, f"{node.index}_heatmap_original_image.png")


# TODO: Use the patch receptive fields instead of upsampling.
def get_closest_patch_imgs(
    image_proto_similarity: ImageProtoSimilarity,
    inverse_transform: Callable[[torch.Tensor], Image],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Gets the pixels for images illustrating the closest patch from an ImageProtoSimilarity.
    Returns: Pixels for: (closest patch, original image, original image with bounding box, original image with heatmap)
    """
    patch_similarities = image_proto_similarity.all_patch_similarities().cpu().numpy()

    # A single pixel is selected.
    # Max because this similarity measure is higher for more similar patches.
    closest_patch_latent_mask = np.uint8(patch_similarities == patch_similarities.max())
    closest_patch_pixel_mask = latent_to_pixel(closest_patch_latent_mask)
    h_low, h_high, w_low, w_high = covering_rectangle_indices(closest_patch_pixel_mask)

    im_original_unscaled = inverse_transform(image_proto_similarity.transformed_image)
    im_original = np.array(im_original_unscaled, dtype=np.float32) / 255

    im_closest_patch = im_original[
        h_low:h_high,
        w_low:w_high,
        :,
    ]

    im_with_bbox = get_im_with_bbox(im_original, h_low, h_high, w_low, w_high)

    pixel_heatmap = latent_to_pixel(patch_similarities)
    colored_heatmap = _to_rgb_map(pixel_heatmap)
    im_with_heatmap = 0.5 * im_original + 0.2 * colored_heatmap

    return im_closest_patch, im_original, im_with_bbox, im_with_heatmap


def covering_rectangle_indices(mask: np.ndarray) -> (int, int, int, int):
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
) -> np.ndarray:
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


def _to_rgb_map(arr: np.ndarray) -> np.ndarray:
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
