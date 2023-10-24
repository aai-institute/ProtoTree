import os

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Callable, List, Union, Literal
from PIL.Image import Image
 
from src.util.image import (
    get_augmentation_transform,
    get_base_transform,
    get_normalize_transform,
)
from src.util.image import MODIFICATIONS
literal_mods = list(MODIFICATIONS.keys())

def get_dataloader(
    dataset_dir: str, img_size: (224, 224), augment: bool = False, train : bool = False, explain=False, modifications=None, loader_batch_size: int = 64, num_workers: int = 4, shuffle: bool = False
):
    
    dataset = get_data(dir=dataset_dir, img_size=img_size, augment=augment, train=train, explain=explain, modifications=modifications)
    return DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers, 
    )
    

def get_data(
    dir, augment=False, train=False, explain=False, modifications=None, img_size=(224, 224)
) -> ImageFolder:
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
    if train:
        data_set = PrototypesExplanationFolder(dir, transform=transform)
    if explain: 
        if modifications is None:
            raise ValueError("Need to select image modifications to explain prototypes")
        data_set = PrototypesExplanationFolder(dir, transform=transform, pipeline_modifications=modifications)
    else:
        if train:
            data_set = PrototypesExplanationFolder(dir, transform=transform, get_img_path=True)
        else:
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


class PrototypesExplanationFolder(ImageFolder):
    """
    This class inherits from ImageFolder.
    return : sample of image, label, path of the image and a list of image modifications 
    """
    def __init__(self, root_dir: str, transform: Callable, pipeline_modifications: Literal[literal_mods] = None, get_img_path: bool = False): 
        super(PrototypesExplanationFolder, self).__init__(root=root_dir)
        
        self.base_transform = transform
        self.get_img_path = get_img_path 
        if pipeline_modifications:
            self.pipeline_modifications = {mod: MODIFICATIONS[mod] for mod in pipeline_modifications}
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index: int): 
        x, y = super(PrototypesExplanationFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        
        if self.get_img_path:
            return (self.base_transform(x), y, path)
        
        x_mods = {mod: self.base_transform(func(x)) for mod, func in self.pipeline_modifications.items()}
        x = self.base_transform(x)  
        
        return (x, y, path, x_mods)