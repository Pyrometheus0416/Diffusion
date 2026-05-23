from dataclasses import dataclass
from typing import Generator, Iterable, TypeAlias, Callable
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg

from config import DEVICE, DTYPE

#--------------------------------------------------------------------
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

#--------------------------------------------------------------------
@dataclass
class AnimeFaceDataset(Dataset):
    floder: Path
    size: int = 80

    mean = (0.6882, 0.5888, 0.5723)
    std = (0.2813, 0.2822, 0.2626)
    # mean = (0.7428, 0.6926, 0.7218)  # Quan_AnimeFace
    # std =  (0.2652, 0.2792, 0.2578)  # Quan_AnimeFace
    inv_std = tuple(1/std_i for std_i in std)
    inv_mean = tuple(-istdi*meani for istdi,meani in zip(inv_std,mean))

    orignal_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, (0.9,1.0), (6/7,7/6)),
        # transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToDtype(torch.float32,scale=True),
        # transforms.GaussianNoise(),
        transforms.Normalize(mean, std),
    ])  # the default transform for training data
    transform: transforms.Compose = orignal_transform

    orignal_inv_trans = transforms.Compose([
        transforms.Normalize(inv_mean,inv_std),
        transforms.Lambda(lambda x:torch.clamp(x,min=0.0,max=1.0)),
        transforms.ToDtype(torch.uint8,scale=True)
    ])  # the default transform for converting tensor to image (for preview)
    inv_trans: transforms.Compose = orignal_inv_trans

    def __post_init__(self):
        self.path: tuple[Path,...] = tuple(self.floder.iterdir())

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        img_path = self.path[index]
        img_t = decode_image(img_path, "RGB").to(device=DEVICE)
        return self.transform(img_t)
    
    def reset(self):
        """reset the transform to the default one (for training data and preview)"""
        self.transform = self.orignal_transform
        self.inv_trans = self.orignal_inv_trans


@dataclass
class COCO2014(Dataset):
    main_floder: Path  # include train2014, val2014, captions2014
    train : bool = True
    base_size: int = 80

    mean = (0.471, 0.448, 0.408)
    std = (0.234, 0.239, 0.242)
    inv_std = tuple(1/std_i for std_i in std)
    inv_mean = tuple(-istdi*meani for istdi,meani in zip(inv_std,mean))

    orignal_transform = transforms.Compose([
        transforms.RandomResizedCrop(base_size, (0.9,1.0), (6/7,7/6)),
        # transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToDtype(torch.float32,scale=True),
        # transforms.GaussianNoise(),
        transforms.Normalize(mean, std),
    ])  # the default transform for training data
    transform: transforms.Compose = orignal_transform

    orignal_inv_trans = transforms.Compose([
        transforms.Normalize(inv_mean,inv_std),
        transforms.Lambda(lambda x:torch.clamp(x,min=0.0,max=1.0)),
        transforms.ToDtype(torch.uint8,scale=True)
    ])  # the default transform for converting tensor to image (for preview)
    inv_trans: transforms.Compose = orignal_inv_trans

    def __post_init__(self):
        target = 'train' if self.train else 'val'

        self.floder = self.main_floder / f"{target}2014"  # include 82783 train images and 40504 val images
        self.annotation_floder = self.main_floder / "captions2014" / 'annotations'
        self.annotations_path = self.annotation_floder / f"captions_{target}2014.json"
        # json → {info, images, licenses, annotations}
        # images → N × {file_name, id, license, coco_url, height, width, date_captured, flickr_url}
        # annotations → N × {image_id, id, caption}
        with open(self.annotations_path, 'r') as f:
            self.annotations: dict = json.load(f)

        img_annotation: list[dict] = self.annotations["images"]
        img_annotation.sort(key=lambda x: x['id'])
        cap_annotation: list[dict] = self.annotations["annotations"]
        cap_annotation.sort(key=lambda x: x['image_id'])

    def __len__(self):
        # assert len(self.annotations['images']) == len(self.annotations['annotations'])
        return len(self.annotations['annotations'])
    
    def __getitem__(self, index):
        #  the image and its caption have the same index, because they are sorted by image_id
        cap: str = self.annotations['annotations'][index]['caption']
        img_path: Path = self.floder / self.annotations['images'][index]['file_name']
        img = decode_image(img_path, "RGB").to(device=DEVICE)
        return self.transform(img), cap
    
    def __str__(self):
        return str(self.annotations['info'])
    
    def reset(self):
        """reset the transform to the default one (for training data and preview)"""
        self.transform = self.orignal_transform
        self.inv_trans = self.orignal_inv_trans
        self.base_size = 80


if __name__ == "__main__":
    src = Path(r"E:\CodeHub\Mydata\COCO")
    dataset = COCO2014(src, train=True)