import numpy as np
from quadtree.quadtree_node import QuadtreeNode


def create_binary_mask(root: QuadtreeNode, image_size: tuple[int, int]) -> np.ndarray:
    """
    Creates a binary mask (numpy array) of the quadtree's leaf regions.
    Pixels inside a leaf region are set to 1, others to 0.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for leaf in root.get_all_leaves():
        x, y, w, h = leaf.get_bounds()
        mask[y:y+h, x:x+w] = 1
    return mask