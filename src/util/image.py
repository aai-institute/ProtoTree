from typing import Callable

import cv2
import numpy as np
import torch
from PIL.Image import Image
from torchvision import transforms as transforms


# TODO: Probably shouldn't need to hardcode these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_augmentation_transform() -> Callable[[torch.Tensor], Image]:
    return transforms.RandomOrder(
        [
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
        ]
    )


def get_inverse_base_transform(
    img_size: tuple[int, int]
) -> Callable[[torch.Tensor], Image]:
    return transforms.Compose(
        [
            get_inverse_normalize_transform(),
            transforms.ToPILImage(),
            transforms.Resize(size=img_size),
        ]
    )


def get_inverse_arr_transform(
    img_size: tuple[int, int]
) -> Callable[[torch.Tensor], np.ndarray]:
    inv_transform = get_inverse_base_transform(img_size)
    return lambda tensor: np.array(inv_transform(tensor), dtype=np.float32) / 255


def get_base_transform(img_size: tuple[int, int]) -> Callable[[torch.Tensor], Image]:
    return transforms.Compose(
        [
            transforms.Resize(size=img_size),
            transforms.ToTensor(),
            get_normalize_transform(),
        ]
    )


def get_inverse_normalize_transform() -> Callable[[torch.Tensor], Image]:
    return transforms.Normalize(
        mean=[-m / s for m, s in zip(_MEAN, _STD)],
        std=[1 / s for s in _STD],
    )


def get_normalize_transform() -> Callable[[torch.Tensor], Image]:
    return transforms.Normalize(mean=_MEAN, std=_STD)


def get_latent_to_pixel(
    img_size: tuple[int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    def latent_to_pixel(latent_img: np.ndarray):
        return cv2.resize(
            latent_img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC
        )
        
    return latent_to_pixel