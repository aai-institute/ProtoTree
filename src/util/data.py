from typing import Callable

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import project_dir, test_dir, train_dir


def get_dataloaders(
    pin_memory=True, batch_size=64, **kwargs
) -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    """

    :param pin_memory:
    :param batch_size:
    :param kwargs: passed to DataLoader
    :return:
    """
    train_set, project_set, test_set = get_data()

    def get_loader(dataset: ImageFolder, batch_size=batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            **kwargs
        )

    train_loader = get_loader(train_set)
    # make batch size smaller to prevent out of memory errors during projection
    project_loader = get_loader(project_set, batch_size=batch_size // 4)
    test_loader = get_loader(test_set)
    return train_loader, project_loader, test_loader


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_normalize_transform():
    return transforms.Normalize(mean=_MEAN, std=_STD)


def get_inverse_normalize_transform():
    return transforms.Normalize(
        mean=[-m / s for m, s in zip(_MEAN, _STD)],
        std=[1 / s for s in _STD],
    )


def get_base_transform(img_size=(224, 224)):
    return transforms.Compose(
        [
            transforms.Resize(size=img_size),
            transforms.ToTensor(),
            get_normalize_transform(),
        ]
    )


def get_inverse_base_transform(img_size=(224, 224)) -> Callable[[torch.Tensor], Image]:
    return transforms.Compose(
        [
            get_inverse_normalize_transform(),
            transforms.ToPILImage(),
            transforms.Resize(size=img_size),
        ]
    )


def get_augmentation_transform():
    return transforms.RandomOrder(
        [
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
        ]
    )


def get_data(
    augment_train_set=True, img_size=(224, 224)
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """

    :param augment_train_set: only affects the train set, project set and test set are not augmented
    :param img_size:
    :return: tuple of type train_set, project_set, test_set, classes, shape
    """
    base_transform = get_base_transform(img_size=img_size)

    if augment_train_set:
        # TODO: why first augment and then resize?
        train_transform = transforms.Compose(
            [
                get_augmentation_transform(),
                transforms.Resize(size=img_size),
                transforms.ToTensor(),
                get_normalize_transform(),
            ]
        )
    else:
        train_transform = base_transform

    # TODO: relax hard-configured datasets, make this into a generic loader
    # TODO 2: we actually train on the corners, why? Is this to reveal biases?
    train_set = ImageFolder(train_dir, transform=train_transform)
    project_set = ImageFolder(project_dir, transform=base_transform)
    test_set = ImageFolder(test_dir, transform=base_transform)
    return train_set, project_set, test_set
