import cv2
import numpy as np
from xycut.separator_model import SeparatorModel
from typing import List
from xycut.block import Block

class SeparatorExtractor:
    def extract(self, block: Block, image: np.ndarray, use_line_separators=True) -> List[SeparatorModel]:
        separators = []

        # --- White space detection ---
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        height, width = image.shape[:2]
        threshold_h = width * 0.9
        threshold_v = height * 0.9

        row_sums = np.sum(closed == 0, axis=1)
        col_sums = np.sum(closed == 0, axis=0)

        # Horizontal whitespace bands
        for y in range(1, height - 1):
            if row_sums[y] > threshold_h and row_sums[y - 1] <= threshold_h:
                h = 2
                sep = SeparatorModel(y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                sep.bounds = (0, y, width, h)
                separators.append(sep)

        # Vertical whitespace bands
        for x in range(1, width - 1):
            if col_sums[x] > threshold_v and col_sums[x - 1] <= threshold_v:
                w = 2
                sep = SeparatorModel(x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                sep.bounds = (x, 0, w, height)
                separators.append(sep)

        # --- Optional line detection ---
        if use_line_separators:
            line_separators = self.extract_line_separators(image, block)
            separators.extend(line_separators)

        return separators

    def extract_line_separators(self, image: np.ndarray, block: Block) -> List[SeparatorModel]:
        separators = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            sep = SeparatorModel(y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
            sep.bounds = (x, y, w, h)
            separators.append(sep)

        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            sep = SeparatorModel(x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
            sep.bounds = (x, y, w, h)
            separators.append(sep)

        return separators


