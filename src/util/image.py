from enum import Enum
from typing import Callable, Union

import cv2
import numpy as np
import PIL
import torch
from PIL.Image import Image
from torchvision import transforms as transforms

# TODO: Probably shouldn't need to hardcode these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# TODO: gio: understand how to put and pass these param from config.cfg
BRIGHT_LVL = 0.8
CONTR_LVL = 0.45
SAT_LVL = 0.7
HUE_LVL = 0.1
AMPLITUDE = 4
WAVE = 0.05
TEXTURE_H = 4

# Transformation definition for explaining prototypes
HUE_TRANSFORM = transforms.ColorJitter(
    brightness=[BRIGHT_LVL, BRIGHT_LVL], contrast=0, saturation=0, hue=0
)
CONTRAST_TRANSFORM = transforms.ColorJitter(
    brightness=0, contrast=[CONTR_LVL, CONTR_LVL], saturation=0, hue=0
)
SATURATION_TRANSFORM = transforms.ColorJitter(
    brightness=0, contrast=0, saturation=[SAT_LVL, SAT_LVL], hue=0
)
COLOR_TRANSFORM = transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=[HUE_LVL, HUE_LVL]
)


class TextureTransformation:
    def __init__(
        self, texture_h: int, templ_window_size: int = 7, search_window_size: int = 7
    ):
        self.texture_h = texture_h
        self.templ_window_size = templ_window_size
        self.search_window_size = search_window_size

    def __call__(self, img: Union[Image, np.ndarray]):
        if type(img) == Image:
            img = np.array(img)

        img_texture = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            templateWindowSize=self.templ_window_size,
            searchWindowSize=self.search_window_size,
            h=self.texture_h,
            hColor=self.texture_h,
        )
        return PIL.Image.fromarray(img_texture)


class ShapeTransformation:
    def __init__(self, amplitude, wave):
        self.amplitude = amplitude
        self.wave = wave

    def __call__(self, img: Union[Image, np.ndarray]):
        if type(img) == Image:
            img = np.array(img)

        shift = lambda x: self.amplitude * np.sin(np.pi * x * self.wave)

        for i in range(img.shape[0]):
            img[i, :, :] = np.roll(img[i, :, :], int(shift(i)), axis=0)
        for i in range(img.shape[1]):
            img[:, i, :] = np.roll(img[:, i, :], int(shift(i)), axis=0)

        return PIL.Image.fromarray(img)


class RegisteredImageTransform(Enum):
    HUE = "hue"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    COLOR = "color"
    SHAPE = "shape"
    TEXTURE = "texture"

    def get_transform(self):
        match self:
            case RegisteredImageTransform.HUE:
                return HUE_TRANSFORM
            case RegisteredImageTransform.CONTRAST:
                return CONTRAST_TRANSFORM
            case RegisteredImageTransform.SATURATION:
                return SATURATION_TRANSFORM
            case RegisteredImageTransform.COLOR:
                return COLOR_TRANSFORM
            case RegisteredImageTransform.SHAPE:
                return ShapeTransformation(amplitude=AMPLITUDE, wave=WAVE)
            case RegisteredImageTransform.TEXTURE:
                return TextureTransformation(texture_h=TEXTURE_H)


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
