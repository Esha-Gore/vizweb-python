import numpy as np
from typing import Callable
from quadtree.quadtree_node import QuadtreeNode
import cv2
import os
os.makedirs("output", exist_ok=True)

#from pdb import set_trace as st
class Quadtree:
    def __init__(self, image: np.ndarray, max_depth: int, min_size: int,
                 entropy_func: Callable[[np.ndarray], float], entropy_threshold: float):
        self.image = image
        self.max_depth = max_depth
        self.min_size = min_size
        self.entropy_func = entropy_func
        self.entropy_threshold = entropy_threshold
        self.step_num = 0

    def build(self, root: QuadtreeNode, depth: int = 0):
        x, y, w, h = root.get_bounds()
        region = self.image[y:y+h, x:x+w]

        entropy = self.entropy_func(region)
        root.set_entropy(entropy)
        if (depth >= self.max_depth or
            w <= self.min_size or h <= self.min_size or
            entropy < self.entropy_threshold):
            return  # stop subdividing
        
        root.subdivide()
        #self._visualize_split((x, y, w, h))
        for child in root.children:
            self.build(child, depth + 1)

    def _visualize_split(self, bounds):
        x, y, w, h = bounds
        vis = self.image.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite(f"output/step_{self.step_num:03d}.png", vis)
        self.step_num += 1

    def get_all_leaves(self, root: QuadtreeNode):
        return root.get_all_leaves()