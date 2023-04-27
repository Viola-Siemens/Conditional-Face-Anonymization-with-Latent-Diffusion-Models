import os
from typing import List, Tuple

import numpy
from PIL import Image
from torch.utils.data import Dataset


class MinimalDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.label_name = {"城市": 0, "乡村": 1}
        self.images = self.get_images(path)
        self.transform = transform

    def __getitem__(self, item) -> Tuple[numpy.ndarray, int]:
        path_img, label = self.images[item]
        img = Image.open(path_img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def get_images(self, path: str) -> List:
        images = []

        for sub_dir in ["城市", "乡村"]:
            train_path = os.path.join(path, sub_dir, "学习")

            imgs_train = list(filter(
                lambda x: x.endswith(".bmp"),
                os.listdir(train_path)
            ))

            for im in imgs_train:
                path_img = os.path.join(train_path, im)
                label = self.label_name[sub_dir]
                images.append((path_img, int(label)))

        return images
