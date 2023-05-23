import os

import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import val_dir, test_dir, train_dir
from util.image import (
    get_augmentation_transform,
    get_base_transform,
    get_normalize_transform,
)


def get_dataloaders(
    pin_memory=True, batch_size=64, **kwargs
) -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    """
    :param pin_memory:
    :param batch_size:
    :param kwargs: passed to DataLoader
    :return:
    """
    train_set, val_set, test_set = get_data()

    def get_loader(dataset: ImageFolder, loader_batch_size=batch_size):
        return DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            **kwargs
        )

    train_loader = get_loader(train_set)
    val_loader = get_loader(val_set)
    test_loader = get_loader(test_set)
    return train_loader, val_loader, test_loader


def get_data(
    augment_train_set=True, img_size=(224, 224)
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    :param augment_train_set: only affects the train set, val set and test set are not augmented
    :param img_size:
    :return: tuple of type train_set, val_set, test_set, classes, shape
    """
    base_transform = get_base_transform(img_size)

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
    val_set = ImageFolder(val_dir, transform=base_transform)
    test_set = ImageFolder(test_dir, transform=base_transform)
    return train_set, val_set, test_set


def save_img(img: np.ndarray, filepath: os.PathLike):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        filepath,
        img,
        vmin=0.0,
        vmax=1.0,
    )
