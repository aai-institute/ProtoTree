import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from prototree.node import InternalNode
from prototree.project import ProjectionPatchInfo
from util.data import get_inverse_base_transform


# adapted from protopnet
# TODO: refactor, use the feature pixels visual field instead of upsampling to some size
@torch.no_grad()
def save_prototype_visualizations(
    node_to_patch_info: dict[InternalNode, ProjectionPatchInfo],
    save_dir: os.PathLike,
    upsample_threshold=0.98,
):
    save_dir = Path(save_dir)
    inverse_transform = get_inverse_base_transform()

    for node, patch_info in node_to_patch_info.items():
        x = inverse_transform(patch_info.image)
        x.save(save_dir / f"{node.index}_original_image.png")

        x_np = np.asarray(x)
        img_size = x_np.shape[:2]

        x_np = np.float32(x_np) / 255
        # TODO: are there grayscale images?
        if x_np.ndim == 2:  # convert grayscale to RGB
            x_np = np.stack((x_np,) * 3, axis=-1)

        similarity_map = torch.exp(-patch_info.all_patch_distances).cpu().numpy()
        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(
            np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET
        )
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[..., ::-1]
        plt.imsave(
            fname=save_dir / f"{node.index}_heatmap_latent_similaritymap.png",
            arr=similarity_heatmap,
            vmin=0.0,
            vmax=1.0,
        )

        upsampled_act_pattern = cv2.resize(
            similarity_map,
            dsize=(img_size[1], img_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET
        )
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]

        overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
        plt.imsave(
            fname=save_dir / f"{node.index}_heatmap_original_image.png",
            arr=overlayed_original_img,
            vmin=0.0,
            vmax=1.0,
        )

        # a single pixel is selected
        feature_patches_mask = (
            patch_info.all_patch_distances <= patch_info.closest_patch_distance
        )
        # and blown up with cubic interpolation to the desired size
        pixels_mask = cv2.resize(
            feature_patches_mask,
            dsize=(img_size[1], img_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        plt.imsave(
            fname=save_dir / f"{node.index}_masked_upsampled_heatmap.png",
            arr=pixels_mask,
            vmin=0.0,
            vmax=1.0,
        )

        high_act_patch_indices = find_high_activation_crop(
            pixels_mask, upsample_threshold
        )
        high_act_patch = x_np[
            high_act_patch_indices[0] : high_act_patch_indices[1],
            high_act_patch_indices[2] : high_act_patch_indices[3],
            :,
        ]
        plt.imsave(
            fname=save_dir / f"{node.index}_nearest_patch_of_image.png",
            arr=high_act_patch,
            vmin=0.0,
            vmax=1.0,
        )

        # save the original image with bounding box showing high activation patch
        imsave_with_bbox(
            fname=save_dir / f"{node.index}_bounding_box_nearest_patch_of_image.png",
            img_rgb=x_np,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3],
            color=(0, 255, 255),
        )
    return node_to_patch_info


# copied from protopnet
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


# copied from protopnet
def imsave_with_bbox(
    fname: os.PathLike,
    img_rgb: np.ndarray,
    bbox_height_start: int,
    bbox_height_end: int,
    bbox_width_start: int,
    bbox_width_end: int,
    color=(0, 255, 255),
):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        img_bgr_uint8,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=2,
    )
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    plt.imsave(fname, img_rgb_float)
