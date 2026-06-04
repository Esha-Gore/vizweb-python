import cv2
import os
import numpy as np
from quadtree.completed_quadtree_feature_computer import FullQuadtreeFeatureComputer
from quadtree.quadtree_visualizer import draw_all_blocks
from quadtree.quadtree_node import QuadtreeNode
from quadtree.quadtree import Quadtree

image_folder = "distinct_images"
output_folder = "output/quadtree"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    # Feature extraction
    qt_computer = FullQuadtreeFeatureComputer()
    qt_computer.compute_features(image)
    features = qt_computer.get_features()

    print(f"\nImage: {filename}")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    # Visualization
    qt = Quadtree(image, max_depth=20, min_size=20,
                  entropy_func=qt_computer.decomposer.strategy.compute_entropy,
                  entropy_threshold=2)
    root = QuadtreeNode(0, 0, image.shape[1], image.shape[0])
    qt.build(root)

    print(f"  Num Leaves: {root.num_leaves()}")

    visualized = draw_all_blocks(image, root)
    cv2.imwrite(os.path.join(output_folder, f"{filename}.png"), visualized)
    print(f"  Saved visualization to {output_folder}/{filename}.png")