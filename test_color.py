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

# Solid Image tests
def make_solid_color_tensor(rgb, size=(4, 4)):
    return torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(3, *size) / 255.0

image_tensor = make_solid_color_tensor((255, 0, 0))

#print("Solid Image Tests:\n")

cf1 = compute_colorfulness(image_tensor)
#print("Colorfulness (Hassler and Susstunk metric):", cf1)

cf2 = compute_colorfulness2(image_tensor)
#print("Colorfulness (LUV metric):", cf2)

avg_hsv = compute_average_hsv(image_tensor)
#print("Average HSV:", avg_hsv.tolist())

color_dist = compute_color_distribution(image_tensor)
#print("Color Distribution:")
#for color, proportion in color_dist.items():
    #print(f"  {color}: {proportion:.2f}")


# Real Image Tests
def load_image_tensor(path):

    image = Image.open(path).convert("RGB")
    transform = T.ToTensor()
    return transform(image)



# Folder with real-world images
image_folder = "distinct_images"

test_number = 2
os.makedirs(f"debug/cm", exist_ok=True)
os.makedirs(f"debug/cm/test_{test_number}", exist_ok=True)

for filename in os.listdir(image_folder):
    #print(f"\nTesting {filename}")
    image_tensor = load_image_tensor(os.path.join(image_folder, filename))

    cf1 = compute_colorfulness(image_tensor)
    cf2 = compute_colorfulness2(image_tensor)
    cf22 = compute_colorfulness22(image_tensor)
    avg_hsv = compute_average_hsv(image_tensor)
    color_dist = compute_color_distribution(image_tensor)

    with open(f"debug/cm/test_{test_number}/log.txt", "a") as f:
        f.write(f"Image {filename}:\n")
        f.write(f"Colorfulness (Hasler and Suesstrunk metric): {cf1}\n")
        f.write(f"Colorfulness (LUV metric) clamping: {cf2}\n")
        f.write(f"Colorfulness (LUV metric) not clamping: {cf22}\n")
        f.write(f"Average HSV: {avg_hsv.tolist()} \n")
        f.write(f"Color Distribution not included \n\n")

    ##print("Colorfulness (Hasler and Suesstrunk metric):", cf1)
    #print("Colorfulness (LUV metric) clamping:", cf2)
    ##print("Colorfulness (LUV metric) not clamping:", cf22)
    ##print("Average HSV:", avg_hsv.tolist())
    ##print("Color Distribution:")
    #for color, proportion in color_dist.items():
    #    #print(f"  {color}: {proportion:.2f}")