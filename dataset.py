from colorfulness_metrics import (
    compute_colorfulness,
    compute_colorfulness2,
    compute_average_hsv,
    compute_color_distribution,
)
from dataclasses import dataclass
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
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
            *[color_dist[c] for c in [nc.name for nc in STANDARD_COLORS]]
        ]

        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)

        return torch.tensor(feature_vector, dtype=torch.float32), label