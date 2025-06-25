import os
import pandas as pd
from torchvision.io import decode_image
from dataclasses import dataclass
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

@dataclass(frozen=True)
class NamedColor:
    name: str
    rgb: tuple  # (R, G, B) range:  0–255

STANDARD_COLORS = [
    NamedColor("red", (255, 0, 0)),
    NamedColor("green", (0, 255, 0)),
    NamedColor("blue", (0, 0, 255)),
    NamedColor("yellow", (255, 255, 0)),
    NamedColor("cyan", (0, 255, 255)),
    NamedColor("magenta", (255, 0, 255)),
    NamedColor("black", (0, 0, 0)),
    NamedColor("white", (255, 255, 255)),
    NamedColor("gray", (128, 128, 128)),
    NamedColor("orange", (255, 165, 0))
]

COLOR_SIMILARITY_THRESHOLD = 0.24 # from vizweb file, bc 200 / 3 * 255 as there are three channels???

def compute_color_distribution(image: torch.Tensor) -> dict[str, float]:
    H, W = image.shape[1], image.shape[2]   # get the height and width of the image
    total_pixels = H * W                    # get the total num of pixels
    distribution = {}                       # intialize the return dict

    for named_color in STANDARD_COLORS:     # for every color in the STANDARD_COLORS
        count = count_pixels_of_color(image, named_color.rgb) # get the number of pixels which match that color
        proportion = count / total_pixels                     # propotion of pixels in whole image which match this color
        distribution[named_color.name] = proportion           # put in dictionary, color as key and proportion as value

    return distribution


def count_pixels_of_color(image: torch.Tensor, target_rgb: tuple) -> int:

    target_color = torch.tensor([c / 255.0 for c in target_rgb], dtype=image.dtype).view(3, 1, 1) #Scales pixel values, converts into Tensor list, reshapes tensor to allow broadcasting
    diff = torch.abs(image - target_color)         # [3, H, W], gets the the absolute difference for each pixel
    diff_sum = diff.sum(dim=0)                     # [H, W], gets the the total difference of RBG values for every pixel across each channel
    mask = diff_sum <= COLOR_SIMILARITY_THRESHOLD  # [H, W], compares to a set threshold
    return int(mask.sum().item())                  # return how many pixels matched.

def compute_average_hsv(image: torch.Tensor) -> torch.Tensor:
    np_image = image.mul(255).byte().permute(1, 2, 0).cpu().numpy()     # make the image compatiable with Open CV
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)               # converting the color space to be in bgr format
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)              # converting the color space to be in HSV format
    hsv_tensor = torch.from_numpy(hsv_image).float().permute(2, 0, 1)   # converts the numpy back into a tensor
    avg_hsv = hsv_tensor.mean(dim=(1, 2))                               # computes the mean of each channel
    return avg_hsv                                                      # returns means


# Hasler and Suesstrunk metrics
def compute_colorfulness(image: torch.Tensor) -> float:
    if image.max() <= 1.0:
        image = image * 255.0
        
    R, G, B = image[0], image[1], image[2]  # Split Channels
    alpha = R - G                           # Compute alpha = |R - G|
    beta = 0.5 * (R + G) - B                # Compute beta = |0.5 * (R + G) - B|

    alpha_mean = alpha.mean()               # Compute means
    beta_mean = beta.mean()

    alpha_std = alpha.std(unbiased=False)   # Compute STD
    beta_std = beta.std(unbiased=False)

    # Apply the colorfulness formula
    colorfulness = torch.sqrt(alpha_std**2 + beta_std**2) + \
                   0.3 * torch.sqrt(alpha_mean**2 + beta_mean**2)

    return colorfulness.item()  # convert from tensor to float


def compute_colorfulness2(image: torch.Tensor) -> float:

    np_image = image.mul(255).byte().permute(1, 2, 0).cpu().numpy()

    luv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2Luv)   # Convert RGB → CIELUV using OpenCV
    L, u, v = cv2.split(luv_image)                          # Each: shape [H, W], values in OpenCV Luv range

    # Convert to PyTorch tensors (float32)
    L = torch.from_numpy(L).float() #* (255.0 / 100.0)
    u = torch.from_numpy(u).float() - 96.0
    v = torch.from_numpy(v).float() - 136.0

    chroma = torch.sqrt(u ** 2 + v ** 2)     # Compute chroma: sqrt(u^2 + v^2)
    saturation = chroma / (L + 1e-6)         # Avoid divide-by-zero: add epsilon to L, CAN REMOVE

    sat_mean = saturation.mean()             # Compute mean + std of saturation   
    sat_std = saturation.std(unbiased=False)

    return (sat_mean + sat_std).item()


def compute_colorfulness22(image: torch.Tensor) -> float:
    # 1) Bring the tensor into H×W×C uint8 [0…255], in RGB order:
    np_uint8 = (image * 255).byte().permute(1, 2, 0).cpu().numpy()

    # 2) Run the exact same OpenCV conversion the Java code used.
    #    (They called CV_BGR2HSV in the snippet, but presumably meant CV_BGR2Luv;
    #     try both if you need to match their output exactly.)
    luv8 = cv2.cvtColor(np_uint8, cv2.COLOR_RGB2Luv)

    # 3) Split out the raw 8-bit channels—with no offset or scaling:
    L8, u8, v8 = cv2.split(luv8)

    print("L8 range:",  L8.min(),  L8.max())
    print("u8 range:",  u8.min(),  u8.max())
    print("v8 range:",  v8.min(),  v8.max())


    # 4) Move into torch and compute saturation exactly like Java:
    L = torch.from_numpy(L8).float()
    u = torch.from_numpy(u8).float()
    v = torch.from_numpy(v8).float()

    L = torch.clamp(L, min=1.0)

    chroma = torch.sqrt(u * u + v * v)
    saturation = chroma / (L + 1e-6)

    # 5) Return mean + std
    return (saturation.mean() + saturation.std(unbiased=False)).item()



# get the top 5 colors from the image to get the color palette
# want the RBG channels and perhaps the percentage of each color in the image
# like a webpage display. 
# test by creating a visualization

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_filename = self.img_labels.iloc[idx, 0]
#         img_path = os.path.join(self.img_dir, img_filename)

#         image = Image.open(img_path).convert("RGB")
#         image_tensor = self.transform(image)

#         cf1 = compute_colorfulness(image_tensor)
#         cf2 = compute_colorfulness2(image_tensor)
#         avg_hsv = compute_average_hsv(image_tensor)
#         color_dist = compute_color_distribution(image_tensor)

#         feature_vector = [
#             cf1,
#             cf2,
#             *avg_hsv.tolist(),
#             *[color_dist[c] for c in [nc.name for nc in STANDARD_COLORS]]
#         ]

#         label = self.img_labels.iloc[idx, 1]
#         if self.target_transform:
#             label = self.target_transform(label)

#         return torch.tensor(feature_vector, dtype=torch.float32), label
    