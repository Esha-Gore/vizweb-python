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

# TTC:
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


def compute_colorfulness22(image: torch.Tensor) -> float:

    np_image = image.mul(255).byte().permute(1, 2, 0).cpu().numpy()

    luv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2Luv)   # Convert RGB → CIELUV using OpenCV
    L, u, v = cv2.split(luv_image)                          # Each: shape [H, W], values in OpenCV Luv range

    # Convert to PyTorch tensors (float32)
    L = torch.from_numpy(L).float() #* (255.0 / 100.0)
    u = torch.from_numpy(u).float() - 96.0
    v = torch.from_numpy(v).float() - 136.0

    print("L8 range:",  L.min(),  L.max())
    print("u8 range:",  u.min(),  u.max())
    print("v8 range:",  v.min(),  v.max())

    chroma = torch.sqrt(u ** 2 + v ** 2)     # Compute chroma: sqrt(u^2 + v^2)
    saturation = chroma / (L + 1e-6)         # Avoid divide-by-zero: add epsilon to L, CAN REMOVE

    sat_mean = saturation.mean()             # Compute mean + std of saturation   
    sat_std = saturation.std(unbiased=False)

    return (sat_mean + sat_std).item()


# scale everything with LUV ranges (Way 1) 
# get max min values and check java convention to see if they did somthing different. 
# returns very close answer, around 12 when expected is 14 for image that ends in 402. 
def compute_colorfulness2(image: torch.Tensor) -> float:
    np_uint8 = (image * 255).byte().permute(1, 2, 0).cpu().numpy()

    luv8 = cv2.cvtColor(np_uint8, cv2.COLOR_RGB2Luv)

    L8, u8, v8 = cv2.split(luv8)
    print("before scaling")
    print("L8 range:",  L8.min(),  L8.max())
    print("u8 range:",  u8.min(),  u8.max())
    print("v8 range:",  v8.min(),  v8.max())


    L = torch.from_numpy(L8).float() * (100.0 / 255)
    u = (torch.from_numpy(u8).float() - 134) * (354/255)
    v = (torch.from_numpy(v8).float() - 140) * (262/255)

    print("after scaling")
    print("L8 range:",  L.min(),  L.max())
    print("u8 range:",  u.min(),  u.max())
    print("v8 range:",  v.min(),  v.max())

    L = torch.clamp(L, min=1.0)
    #lowest L can be is 1, greatest is 100
    
    chroma = torch.sqrt(u * u + v * v)
    # smallest is when l is 100 and chroma is smallest it can be. 
    saturation = chroma / (L) # + 1e-6

    return (saturation.mean() + saturation.std()).item()

def get_top_5_colors(image_tensor):

    pixels = image_tensor.reshape(-1, 3)    # Flatten pixels
    total_pixels = pixels.shape[0]          # Get number of total pixels

    # Get unique colors and their counts
    colors, counts = torch.unique(pixels, dim=0, return_counts=True)

    # Get top 5
    top_indices = torch.topk(counts, k=5).indices
    top_colors = colors[top_indices]
    top_counts = counts[top_indices]

    # Convert to dict with percentages
    result = {}
    for color, count in zip(top_colors, top_counts):
        rgb_tuple = tuple(color.tolist())
        percentage = count.item() / total_pixels
        result[rgb_tuple] = percentage

    return result