import cv2
import os
import numpy as np
from quadtree.completed_quadtree_feature_computer import FullQuadtreeFeatureComputer
from quadtree.quadtree_visualizer import draw_quadtree_blocks, draw_all_blocks
from quadtree.quadtree_node import QuadtreeNode
from quadtree.quadtree import Quadtree

# BAD FIX MEMEMEMEMMEMEMMEME
# Load image
image_path = "images/593525f07e402.jpg"
image1 = cv2.imread(image_path)

# Example dimensions: width=1440, height=900
width, height = 1440, 900

# Create a white image (255 for each RGB channel)
white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
black_image = np.zeros((height, width, 3), dtype=np.uint8)
red_green = np.zeros((height, width, 3), dtype=np.uint8)

red_green[:, :width // 2] = [0, 0, 255]
red_green[:, width // 2:] = [0, 255, 0]
images = []

images.append(white_image)
images.append(black_image)
images.append(red_green)

image_folder = "images"

# for image in images:

#     # Run feature extraction
#     qt_computer = FullQuadtreeFeatureComputer()
#     qt_computer.compute_features(image)
#     features = qt_computer.get_features()

#     # #print feature values
#     #print(f"Features for: basic")
#     for key, value in features.items():
#         #print(f"  {key}: {value:.4f}")

#     # Visualize  the quadtree block outlines
#     qt = Quadtree(image, max_depth=5, min_size=20,
#                 entropy_func=qt_computer.decomposer.strategy.compute_entropy,
#                 entropy_threshold=qt_computer.decomposer.strategy.entropy_threshold)
#     root = QuadtreeNode(0, 0, image.shape[1], image.shape[0])
#     qt.build(root)

#     visualized = draw_quadtree_blocks(image, root)

#     cv2.imshow("Quadtree Visualization", visualized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

for filename in os.listdir(image_folder):
    filename = "5935268355cac.jpg"
    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)
    # Run feature extraction
    qt_computer = FullQuadtreeFeatureComputer()
    qt_computer.compute_features(image)
    features = qt_computer.get_features()

    # #print feature values
    #print(f"Features for: {image_path}")
    for key, value in features.items():
        #print(f"  {key}: {value:.4f}")

    # Visualize  the quadtree block outlines
    # TTC: what's the minimum size and max depth? 
    qt = Quadtree(image, max_depth=20, min_size=10,
                entropy_func=qt_computer.decomposer.strategy.compute_entropy,
                entropy_threshold=qt_computer.decomposer.strategy.entropy_threshold)
    root = QuadtreeNode(0, 0, image.shape[1], image.shape[0])
    qt.build(root)

    visualized = draw_all_blocks(image, root)

    cv2.imshow("Quadtree Visualization", visualized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


