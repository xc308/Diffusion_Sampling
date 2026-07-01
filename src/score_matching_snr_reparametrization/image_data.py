from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(
    *,
    batch_size: int = 128,
    image_size: int = 32,
    data_dir: str = "data",
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for MNIST images.

    The images are resized to image_size x image_size and normalised to [-1, 1].

    Args:
        batch_size: Number of images per batch.
        image_size: Image height and width after resizing.
        data_dir: Folder where MNIST data is stored/downloaded.
        num_workers: Number of extra subprocessor worker processes for data loading.
        shuffle: Whether to shuffle the training data.

    Returns:
        A DataLoader yielding batches of shape (B, 1, image_size, image_size).
    """
    transform = transforms.Compose(
        [ # a list contains the transformation pipline, apply the transformation one by one in this order
            transforms.Resize((image_size, image_size)), # resize from 28 to 32 for easy UNet symmetry
            transforms.ToTensor(),                # turn each image into a tensor (1, 28, 28), 
                                                  # and each pixel value is in [0, 1]
            transforms.Normalize((0.5,), (0.5,)), # transforms.Normalize(mean for each chanel, std for each chanel), 
                                                  # new_pixel = (old_pixel - 0.5) / 0.5
                                                  # if RGB, (0.5, 0.5, 0.5) for both mean and std
                                                  # now each pixel value in [-1, 1]
        ]
    )

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,         # load the 60,000 training image, not 10,000 testing image (train= False)
        download=True,
        transform=transform, # for each image, apply transform pipline 
    )                      # each image used for training is shape (1, 32, 32), pixel value in [-1, 1]

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, 
        drop_last=True,    # If the last batch is smaller than batch_size, skip it. 
                          # total sample image is not diviable for the image_size.
                          # B in the (B, 1, 32, 32) will remain the same for every batch
                          # make training more stable
    )

    return dataloader  # returns an object that can repeatedly give mini-batches of MNIST data