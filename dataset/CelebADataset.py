import os
from typing import List, Tuple

import numpy
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images = self.get_images(path)
        self.transform = transform

    def __getitem__(self, item) -> Tuple[numpy.ndarray, int]:
        path_img, label = self.images[item]
        img = Image.open(path_img).crop((8, 28, 168, 188)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def get_images(self, path: str) -> List:
        images = []

        imgs_train = list(filter(
            #lambda x: x.endswith(".jpg"),
            lambda x: x.endswith(".png"),
            os.listdir(path)
        ))

        for im in imgs_train:
            path_img = os.path.join(path, im)
            images.append((path_img, 0))

        return images
