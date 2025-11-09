import cv2
import os
import numpy as np
from quadtree.completed_quadtree_feature_computer import FullQuadtreeFeatureComputer
from quadtree.quadtree_visualizer import draw_quadtree_blocks, draw_all_blocks
from quadtree.quadtree_node import QuadtreeNode
from quadtree.quadtree import Quadtree

image_folder = "distinct_images"
os.makedirs("quadtree_hyperparameter_tuning", exist_ok=True)
os.makedirs("quadtree_hyperparameter_tuning/entropytake2", exist_ok=True)

values = []

for i in range(0,9):

    et = 3.0 + (0.1 * i)
    os.makedirs(f"quadtree_hyperparameter_tuning/entropytake2/ET={et}", exist_ok=True)
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        image = cv2.imread(image_path)
        # Run feature extraction
        qt_computer = FullQuadtreeFeatureComputer()
        qt_computer.compute_features(image)
        features = qt_computer.get_features()

        # #print feature values
        #print(f"Features for: {image_path}")
        for key, value in features.items():

            #print(f"entropy: {qt_computer.decomposer.strategy.entropy_threshold}")

            qt = Quadtree(image, max_depth=10, min_size=15,
                        entropy_func=qt_computer.decomposer.strategy.compute_entropy,
                        entropy_threshold=et)
            root = QuadtreeNode(0, 0, image.shape[1], image.shape[0])
            qt.build(root)

            visualized = draw_all_blocks(image, root)

            cv2.imwrite(f"quadtree_hyperparameter_tuning/entropytake2/ET={et}/{filename}.png", visualized)
            #print(f"done:{filename}")