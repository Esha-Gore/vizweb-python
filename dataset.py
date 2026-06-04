import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from colorfulness_metrics import (
    compute_colorfulness,
    compute_colorfulness2,
    compute_average_hsv,
    compute_color_distribution,
    STANDARD_COLORS,
)

class CustomImageDataset(Dataset):
    """
    PyTorch Dataset for computing vizweb features over a labeled image dataset.

    Args:
        annotations_file: path to a CSV with columns [filename, label]
        img_dir: folder containing the images
        transform: torchvision transform to apply to each image

    NOTE: currently only computes colorfulness features.
    TODO: add quadtree and XY-cut features to the feature vector.
    """

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        cf1 = compute_colorfulness(image_tensor)
        cf2 = compute_colorfulness2(image_tensor)
        avg_hsv = compute_average_hsv(image_tensor)
        color_dist = compute_color_distribution(image_tensor)

        feature_vector = [
            cf1,
            cf2,
            *avg_hsv.tolist(),
            *[color_dist[c.name] for c in STANDARD_COLORS],
        ]

        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)

        return torch.tensor(feature_vector, dtype=torch.float32), label