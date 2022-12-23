from functools import cache
from pathlib import Path

from config import cub_dir, cub_images_dir

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
