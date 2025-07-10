import cv2
import numpy as np
from typing import List, Tuple
from xycut.block import Block  # assumes you have Block class defined here


class XYTreeVisualizer:
    def __init__(self, thickness: int = 1, block_color=(0, 255, 0), separator_color=(0, 0, 255)):
        # thickness: Line thickness for drawing blocks/separators
        self.thickness = thickness
        self.block_color = block_color
        self.separator_color = separator_color

    def draw_blocks(self, image: np.ndarray, root: Block) -> np.ndarray:
        # Recursively draw all blocks as rectangles on a copy of the image
        img_copy = image.copy()
        self._draw_recursive(img_copy, root)
        return img_copy

    def _draw_recursive(self, img: np.ndarray, block: Block):
        x, y, w, h = block.bounds
        cv2.rectangle(img, (x, y), (x + w, y + h), self.block_color, self.thickness)

        for child in block.get_children():
            self._draw_recursive(img, child)

    def draw_block_outlines(self, image: np.ndarray, root_block: Block) -> np.ndarray:
        img_copy = image.copy()
        self._draw_recursive(img_copy, root_block)
        return img_copy

    def draw_separators(self, image: np.ndarray, separators: List[Tuple[int, int, int, int]]) -> np.ndarray:
        # Draw given separator lines on the image
        # separators: List of (x, y, w, h)
        img_copy = image.copy()
        for x, y, w, h in separators:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), self.separator_color, self.thickness)
        return img_copy

    def draw_blocks_and_separators(self, image: np.ndarray, root: Block, separators: List[Tuple[int, int, int, int]]) -> np.ndarray:
        # Draw both blocks and separators
        img_copy = image.copy()
        self._draw_recursive(img_copy, root)
        for x, y, w, h in separators:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), self.separator_color, self.thickness)
        return img_copy
