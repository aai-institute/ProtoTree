import os

import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import val_dir, test_dir, train_dir
from src.util.image import (
    get_augmentation_transform,
    get_base_transform,
    get_normalize_transform,
)


def get_dataloader(
    dataset_dir: str, augment: bool = False, loader_batch_size: int = 64, num_workers: int = 4, shuffle: bool = False
):
    
    dataset = get_data(dir=dataset_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers, 
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
    train_set = get_data(dir=train_dir, augment=True)
    val_set = get_data(dir=val_dir)
    test_set = get_data(dir=test_dir)

    def get_loader(
        dataset: ImageFolder, loader_batch_size: int = batch_size, shuffle: bool = False
    ):
        return DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **kwargs
        )

    train_loader = get_loader(train_set, shuffle=True)
    val_loader = get_loader(val_set)
    test_loader = get_loader(test_set)
    return train_loader, val_loader, test_loader


def get_data(
    dir, augment=False, img_size=(224, 224)
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    :param augment_train_set: only affects the train set, val set and test set are not augmented
    :param img_size:
    :return: tuple of type train_set, val_set, test_set, classes, shape
    """
    base_transform = get_base_transform(img_size)

    if augment:
        # TODO: why first augment and then resize?
        transform = transforms.Compose(
            [
                get_augmentation_transform(),
                transforms.Resize(size=img_size),
                transforms.ToTensor(),
                get_normalize_transform(),
            ]
        )
    else:
        transform = base_transform

    # TODO: relax hard-configured datasets, make this into a generic loader
    # TODO 2: we actually train on the corners, why? Is this to reveal biases?
    data_set = ImageFolder(dir, transform=transform)   
   
    return data_set

def save_img(img: np.ndarray, filepath: os.PathLike):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        filepath,
        img,
        vmin=0.0,
        vmax=1.0,
    )
