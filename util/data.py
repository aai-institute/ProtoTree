from typing import Literal

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import project_dir, test_dir, train_dir


def get_data(
    dataset: Literal["CUB", "CARS"] = "CUB", augment_train_set=True
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    Load the proper dataset based on the parsed arguments
    :return: tuple consisting of:
        - The train data set
        - The project data set (usually train data set without augmentation)
        - The test data set
    """
    if dataset == "CUB":
        return get_birds(augment_train_set=augment_train_set)
    if dataset == "CARS":
        raise NotImplementedError()
    raise ValueError(f'Could not load "{dataset=}"')


def get_dataloaders(
    dataset: Literal["CUB", "CARS"] = "CUB", disable_cuda=False, batch_size=64
) -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    """
    Get data loaders
    """
    train_set, project_set, test_set = get_data(dataset)
    pin_memory = not disable_cuda and torch.cuda.is_available()

    def get_loader(dataset: ImageFolder, batch_size=batch_size):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
        )

    train_loader = get_loader(train_set)
    # make batch size smaller to prevent out of memory errors during projection
    project_loader = get_loader(project_set, batch_size=batch_size // 4)
    test_loader = get_loader(test_set)
    return train_loader, project_loader, test_loader


def get_birds(
    augment_train_set=True, img_size=224
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """

    :param augment_train_set: only affects the train set, project set and test set are not augmented
    :param img_size:
    :return: tuple of type train_set, project_set, test_set, classes, shape
    """
    # TODO 2: we actually train on the corners, why? Is this to reveal biases?

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    base_transform = transforms.Compose(
        [transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), normalize]
    )
    if augment_train_set:
        # TODO: why first augment and then resize?
        transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomOrder(
                    [
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.ColorJitter(
                            (0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(
                            degrees=10, shear=(-2, 2), translate=[0.05, 0.05]
                        ),
                    ]
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = base_transform

    # TODO: relax hard-configured datasets, make this into a generic loader
    train_set = ImageFolder(train_dir, transform=transform)
    project_set = ImageFolder(project_dir, transform=base_transform)
    test_set = ImageFolder(test_dir, transform=base_transform)
    return train_set, project_set, test_set
