"""
This script creates a dataset of cropped images from the CUB dataset. It currently copies a lot of stuff around
and is thus very inefficient in terms of storage. In a future version we might want to assemble the dataset on the fly.
"""

import logging
import shutil
from functools import cache
from pathlib import Path
from typing import Literal

from PIL import Image
from tqdm import tqdm

from src.config import cub_dir, cub_images_dir, dataset_dir

train_crop_dir = dataset_dir / "train_crop"
train_corners_dir = dataset_dir / "train_corners"
test_full_dir = dataset_dir / "test_full"
test_crop_dir = dataset_dir / "test_crop"


log = logging.getLogger(__name__)


def get_image_basedir_name_tuple(image_path: str) -> tuple[str, str]:
    return tuple(*image_path.split("/"))


def save_image(image: Image, path: Path):
    log.debug(f"Saving image to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


cub_images_txt = cub_dir / "images.txt"
cub_split_indices_txt = cub_dir / "train_test_split.txt"
cub_bounding_boxes = cub_dir / "bounding_boxes.txt"
cub_labels = cub_dir / "image_class_labels.txt"


@cache
def get_image_id_path_dict() -> dict[int, Path]:
    """
    Mapping image ids to their path (relative to the root dir).
    """
    image_id_path_dict = {}
    with open(cub_images_txt, "r") as f:
        for line in f:
            image_id, image_subpath = line.split()
            image_id_path_dict[int(image_id)] = cub_images_dir / image_subpath
    return image_id_path_dict


@cache
def get_image_id_train_test_dict() -> dict[int, bool]:
    """
    Mapping image_id to whether it is a train image or not.
    """
    image_id_train_test_dict = {}
    with open(cub_split_indices_txt, "r") as f:
        for line in f:
            image_id, is_train = line.split()
            image_id_train_test_dict[int(image_id)] = bool(int(is_train))
    return image_id_train_test_dict


@cache
def get_image_id_bbox_dict() -> dict[int, tuple[int, int, int, int]]:
    """
    Mapping image ids to bounding boxes. The bounding boxes are tuples of (left, top, right, bottom).
    """
    image_id_bbox_dict = {}
    with open(cub_bounding_boxes, "r") as f:
        for line in f:
            image_id, x, y, w, h = [int(float(i)) for i in line.split()]
            image_id_bbox_dict[image_id] = (x, y, x + w, y + h)
    return image_id_bbox_dict


def save_cropped_cub_images(overwrite: bool = False):
    image_id_path_dict = get_image_id_path_dict()
    image_id_train_test_dict = get_image_id_train_test_dict()
    image_id_bbox_dict = get_image_id_bbox_dict()

    for image_id, is_train in tqdm(
        image_id_train_test_dict.items(), desc="Saving processed images"
    ):
        is_test = not is_train
        image_path = image_id_path_dict[image_id]
        bbox = image_id_bbox_dict[image_id]

        target_crop_dir = train_crop_dir if is_train else test_crop_dir
        image_subpath = Path(image_path.parent.name) / image_path.name
        cropped_image_path = target_crop_dir / image_subpath

        if is_test:
            test_full_image_path = test_full_dir / image_subpath
            if not test_full_image_path.exists() or overwrite:
                log.debug(f"Copying test image to: {test_full_image_path}")
                test_full_image_path.parent.mkdir(parents=True, exist_ok=True)
                # TODO: to we need to convert to RGB, or is copying enough? Why do we need the full images?
                shutil.copy(image_path, test_full_image_path)

        if cropped_image_path.exists() and not overwrite:
            log.info(
                f"Skipping extraction of {cropped_image_path} b/c it already exists. "
                f"NOTE: we also assume that the corners have already been extracted!"
            )
            continue

        # TODO: is the conversion needed?
        image = Image.open(image_path).convert("RGB")
        cropped_image = image.crop(bbox)
        save_image(cropped_image, cropped_image_path)

        if is_train:
            # special treatment for train images:
            # we extract corners (larger than the cropped ones) as training images
            # The center-cropped images are used during the projection as source for the prototypes
            # TODO: unclear whether that's really necessary, why don't we train on the center-cropped images?
            save_corner_subimages(image, bbox, image_path)
            normal_image_path = (
                train_corners_dir / image_path.parent.name / f"normal_{image_path.name}"
            )
            save_image(image, normal_image_path)


def get_corner_bbox(
    orientation_y: Literal["upper", "lower"],
    orientation_x: Literal["left", "right"],
    bbox: tuple[int, int, int, int],
    full_size: tuple[int, int],
    margin_percentage=0.1,
) -> tuple[int, int, int, int]:
    """
    Get a bounding box for the selected corner of an image. The corners go from the selected edge of the image to the
    opposite edge of a "scaled" bounding box. The scaling is done by moving the edges of the bounding box
    outwards by the margin_percentage.

    :param orientation_y:
    :param orientation_x:
    :param bbox: in format left, upper, right, lower
    :param full_size: width, height of the full image. Typically extracted from `image.size`.
        **Note**: this is in the reverse order of `arr.shape`!
    :param margin_percentage: the height and width of the original bounding box will be scaled by this percentage
    :return: bbox of format (x, y, w, h)
    """
    x_max, y_max = full_size
    left, upper, right, lower = bbox
    h = lower - upper
    w = right - left
    hmargin = int(margin_percentage * h)
    wmargin = int(margin_percentage * w)
    if orientation_y == "upper":
        corner_y = 0
        corner_h = min(lower + hmargin, y_max)
    elif orientation_y == "lower":
        corner_y = max(upper - hmargin, 0)
        corner_h = y_max
    else:
        raise ValueError(f"Unknown orientation_y: {orientation_y}")
    if orientation_x == "left":
        corner_x = 0
        corner_w = min(right + wmargin, x_max)
    elif orientation_x == "right":
        corner_x = max(left - wmargin, 0)
        corner_w = x_max
    else:
        raise ValueError(f"Unknown orientation_x: {orientation_x}")
    return corner_x, corner_y, corner_w, corner_h


def save_corner_subimages(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    image_path: Path,
    margin_percentage=0.1,
):
    """
    Save the four corners of the image as separate images. The corners go from the selected edge of the image to the
    opposite edge of a "scaled" bounding box. The scaling is done by moving the edges of the bounding box
    outwards by the margin_percentage.

    :param image:
    :param bbox: in format left, upper, right, lower
    :param image_path:
    :param margin_percentage: the height and width of the original bounding box will be scaled by this percentage.
    :return:
    """
    for orientation_y in ["upper", "lower"]:
        for orientation_x in ["left", "right"]:
            save_image_name = f"{orientation_y}{orientation_x}_{image_path.name}"
            save_image_path = (
                train_corners_dir / image_path.parent.name / save_image_name
            )
            corner_bbox = get_corner_bbox(
                orientation_y, orientation_x, bbox, image.size, margin_percentage
            )

            corner_image = image.crop(corner_bbox)
            save_image(corner_image, save_image_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    save_cropped_cub_images()
