import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def train_data_loader(batch_size = 64, imgz = 128, workers = 0, pin_memory = True):
    # Please specify your own dataset path here
    data_dir = os.path.join('E:/Code_Wenyu/Dataset/ILSVRC2012_img_val')
    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.CenterCrop(imgz),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers,
        pin_memory = pin_memory
    )
    return train_data_loader

def test_data_loader(batch_size = 64, imgz = 128, workers = 0, pin_memory = True):
    # Please specify your own dataset path here
    data_dir = os.path.join('E:/Code_Wenyu/Dataset/test_data')
    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.CenterCrop(imgz),
            transforms.ToTensor(),
        ])
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        pin_memory = pin_memory
    )
    return test_data_loader




