import numpy as np
from quadtree.quadtree_node import QuadtreeNode
from quadtree.quadtree import Quadtree
from quadtree.color_entropy_strategy import ColorEntropyStrategy


class QuadTreeDecomposer:
    def __init__(self, max_depth=5, min_size=20, entropy_threshold=3.0):
        self.max_depth = max_depth
        self.min_size = min_size
        self.strategy = ColorEntropyStrategy(entropy_threshold=entropy_threshold)

    def decompose(self, image: np.ndarray) -> QuadtreeNode:
        height, width = image.shape[:2]
        root = QuadtreeNode(0, 0, width, height)

        qt = Quadtree(
            image=image,
            max_depth=self.max_depth,
            min_size=self.min_size,
            entropy_func=self.strategy.compute_entropy,
            entropy_threshold=self.strategy.entropy_threshold
        )
        qt.build(root)
        return root