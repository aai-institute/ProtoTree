from typing import Literal

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import dataset_dir


# TODO: simplify signature
def get_data(dataset: Literal["CUB", "CARS"] = "CUB", augment_train_set=True):
    """
    Load the proper dataset based on the parsed arguments
    :return: a 5-tuple consisting of:
        - The train data set
        - The project data set (usually train data set without augmentation)
        - The test data set
        - a tuple containing all possible class labels
        - a tuple containing the shape (depth, width, height) of the input images
    """
    if dataset == "CUB":
        return get_birds(augment_train_set=augment_train_set)
    if dataset == "CARS":
        raise NotImplementedError()
        # return get_cars(
        #     True,
        #     "./data/cars/dataset/train",
        #     "./data/cars/dataset/train",
        #     "./data/cars/dataset/test",
        # )
    raise ValueError(f'Could not load "{dataset=}"')


def get_dataloaders(
    dataset: Literal["CUB", "CARS"] = "CUB", disable_cuda=False, batch_size=64
):
    """
    Get data loaders
    """
    # Obtain the dataset
    train_set, project_set, test_set, classes, shape = get_data(dataset)
    c, w, h = shape
    # Determine if GPU should be used
    pin_memory = not disable_cuda and torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    project_loader = DataLoader(
        project_set,
        # make batch size smaller to prevent out of memory errors during projection
        batch_size=batch_size // 4,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    print("Num classes (k) = ", len(classes), flush=True)
    return train_loader, project_loader, test_loader, classes, c


def get_birds(augment_train_set=True, img_size=224) -> tuple:
    """

    :param augment_train_set: only affects the train set, project set and test set are not augmented
    :param img_size:
    :return: tuple of type train_set, project_set, test_set, classes, shape
    """
    # TODO: move those to config.py
    # TODO 2: we actually train on the corners, why? Is this to reveal biases?
    train_dir = dataset_dir / "train_corners"
    project_dir = dataset_dir / "train_crop"
    test_dir = dataset_dir / "test_full"

    shape = (3, img_size, img_size)
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

    train_set = ImageFolder(train_dir, transform=transform)
    project_set = ImageFolder(project_dir, transform=base_transform)
    test_set = ImageFolder(test_dir, transform=base_transform)
    classes = train_set.classes
    for i in range(len(classes)):
        classes[i] = classes[i].split(".")[1]
    return train_set, project_set, test_set, classes, shape


# def get_cars(
#     augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224
# ):
#     shape = (3, img_size, img_size)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#
#     normalize = transforms.Normalize(mean=mean, std=std)
#     transform_no_augment = transforms.Compose(
#         [transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), normalize]
#     )
#
#     if augment:
#         transform = transforms.Compose(
#             [
#                 transforms.Resize(
#                     size=(img_size + 32, img_size + 32)
#                 ),  # resize to 256x256
#                 transforms.RandomOrder(
#                     [
#                         transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
#                         transforms.ColorJitter(
#                             (0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.4, 0.4)
#                         ),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomAffine(degrees=15, shear=(-2, 2)),
#                     ]
#                 ),
#                 transforms.RandomCrop(size=(img_size, img_size)),  # crop to 224x224
#                 transforms.ToTensor(),
#                 normalize,
#             ]
#         )
#     else:
#         transform = transform_no_augment
#
#     trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
#     projectset = torchvision.datasets.ImageFolder(
#         project_dir, transform=transform_no_augment
#     )
#     testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
#     classes = trainset.classes
#
#     return trainset, projectset, testset, classes, shape
