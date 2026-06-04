from colorfulness_metrics import (
    compute_colorfulness,
    compute_colorfulness2,
    compute_colorfulness22,
    compute_average_hsv,
    compute_color_distribution,
)
import torchvision.transforms as T
from PIL import Image
import torch
import os


def make_solid_color_tensor(rgb, size=(4, 4)):
    return torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(3, *size) / 255.0


def load_image_tensor(path):
    image = Image.open(path).convert("RGB")
    return T.ToTensor()(image)


# Solid color sanity check
print("Solid red image:")
image_tensor = make_solid_color_tensor((255, 0, 0))
print("  Colorfulness (Hasler & Suesstrunk):", compute_colorfulness(image_tensor))
print("  Colorfulness (LUV, clamped):", compute_colorfulness2(image_tensor))
print("  Average HSV:", compute_average_hsv(image_tensor).tolist())


# Real image tests
image_folder = "distinct_images"

for filename in os.listdir(image_folder):
    image_tensor = load_image_tensor(os.path.join(image_folder, filename))

    cf1 = compute_colorfulness(image_tensor)
    cf2 = compute_colorfulness2(image_tensor)
    cf22 = compute_colorfulness22(image_tensor)
    avg_hsv = compute_average_hsv(image_tensor)

    print(f"\nImage: {filename}")
    print(f"  Colorfulness (Hasler & Suesstrunk): {cf1}")
    print(f"  Colorfulness (LUV, clamped): {cf2}")
    print(f"  Colorfulness (LUV, unclamped): {cf22}")
    print(f"  Average HSV: {avg_hsv.tolist()}")