import numpy as np
from quadtree.quadtree_node import QuadtreeNode


def create_binary_mask(root: QuadtreeNode, image_size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.uint8)
    for leaf in root.get_all_leaves():
        x, y, w, h = leaf.get_bounds()
        #TTC: The java originally has limits on 30 pixels but I beleve their scaling is different b/c that did not work with ours
        min_w, min_h = image_size[1] // 10, image_size[0] // 10
        if w <= min_w and h <= min_h:
            mask[y:y+h, x:x+w] = 1
    return mask
