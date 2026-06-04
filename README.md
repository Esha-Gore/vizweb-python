# vizweb-python

This repository contains a Python reimplementation of [vizweb](https://github.com/rmardiko/vizweb), a Java library that analyzes UI/UX screenshots by computing visual complexity and colorfulness features.

These metrics and the original repository come from the following paper:
> K. Reinecke et al. (2013). [Predicting users' first impressions of website aesthetics with a quantification of perceived visual complexity and colorfulness](https://dash.harvard.edu/bitstream/handle/1/12561368/Predicting%20Users%20First%20Impressions.pdf). CHI'13.

It computes three categories of features from an input image:
- **Colorfulness** — Hasler & Suesstrunk metric, LUV-based metric, average HSV, color distribution
- **Quadtree decomposition** — horizontal/vertical symmetry and balance, equilibrium, number of leaves
- **XY-cut decomposition** — average/max depth, number of leaves

---

## Environment

- **Python**: 3.11 (tested with 3.11.13)
- **Virtual Environment**: `venv`

---

## Setup

```bash
git clone https://github.com/Esha-Gore/vizweb-python.git
cd vizweb-python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Colorfulness features

```python
from colorfulness_metrics import (
    compute_colorfulness,
    compute_colorfulness2,
    compute_average_hsv,
    compute_color_distribution,
)
import torchvision.transforms as T
from PIL import Image

image = Image.open("your_image.jpg").convert("RGB")
image_tensor = T.ToTensor()(image)

print("Colorfulness (Hasler & Suesstrunk):", compute_colorfulness(image_tensor))
print("Colorfulness (LUV):", compute_colorfulness2(image_tensor))
print("Average HSV:", compute_average_hsv(image_tensor).tolist())
```

### Quadtree features

```python
import cv2
from quadtree.completed_quadtree_feature_computer import FullQuadtreeFeatureComputer

image = cv2.imread("your_image.jpg")

qt_computer = FullQuadtreeFeatureComputer()
qt_computer.compute_features(image)
features = qt_computer.get_features()

for key, value in features.items():
    print(f"  {key}: {value:.4f}")
```

### XY-cut features

```python
import cv2
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer

image = cv2.imread("your_image.jpg")

decomposer = XYDecomposer()
root = decomposer.decompose(image, DefaultXYDecompositionStrategy())

print("Average decomposition depth:", XYFeatureComputer.compute_average_decomposition_level(root))
print("Number of leaf blocks:", XYFeatureComputer.compute_num_leaves(root))
```

---

## Test Scripts

Run these from the repo root. Each processes the images in `distinct_images/` and prints results to stdout. These distinct images are a small subset of diverse webpage screenshots from the original paper's dataset. 

```bash
python test_color.py
python test_quadtree.py
python test_xy.py
```

