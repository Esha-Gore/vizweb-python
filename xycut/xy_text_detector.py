import numpy as np
from xycut.block import Block
from xycut.block_text_detector import BlockTextDetector


class XYTextDetector:
    def __init__(self, root_node: Block, input_image: np.ndarray):
        self.root = root_node
        self.image = input_image

    def detect(self) -> Block:
        self._recursive_detect(self.root, self.image)
        return self.root

    def _recursive_detect(self, node: Block, image: np.ndarray):
        detector = BlockTextDetector(node, image)
        detector.detect()

        for child in node.get_children():
            x, y, w, h = child.bounds
            crop = image[y:y+h, x:x+w]
            self._recursive_detect(child, crop)

