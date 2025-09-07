import numpy as np
import cv2
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block

class LineSeparatorExtractor:
    # Initializes the extractor with debug settings and detection thresholds
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.line_pixel_ratio_threshold = 0.85
        self.ratio_decrement = 0.1
        self.max_retries = 3

    # Applies edge-based logic to extract horizontal and vertical line separators from a block's region
    def extract(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
        separators: List[SeparatorModel] = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny thresholds chosen to match the old Java:
        # cvCanny(gray, mask, 0.66*50, 1.33*50, 3) → (33, 67, aperture=3).
        edges = cv2.Canny(gray, 33, 67, apertureSize=3)

        # Morphological closing to bridge tiny gaps between line pixels.
        # iterations=1 matches Java's single dilate + single erode.
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        height, width = edges.shape

        for retry in range(self.max_retries):
            # todo: check these thesholds against java
            threshold_h = width * (self.line_pixel_ratio_threshold - retry * self.ratio_decrement)
            threshold_v = height * (self.line_pixel_ratio_threshold - retry * self.ratio_decrement)

            row_counts = np.count_nonzero(edges, axis=1)
            col_counts = np.count_nonzero(edges, axis=0)

            horizontal = self._find_verified_separators(row_counts, threshold_h, is_horizontal=True,  block=block, width=width,  height=height)
            vertical   = self._find_verified_separators(col_counts, threshold_v, is_horizontal=False, block=block, width=width,  height=height)

            separators = horizontal + vertical
            if separators:
                break

        return separators

    # Filters and verifies strong line candidates using neighboring pixel density checks
    def _find_verified_separators(self, values, threshold, is_horizontal, block, width=None, height=None):
        verified: List[SeparatorModel] = []
        for i in range(1, len(values) - 1):
            if values[i] > threshold:
                before, current, after = values[i - 1], values[i], values[i + 1]
                if before < 0.2 * current and after < 0.2 * current:
                    if is_horizontal:
                        separator = SeparatorModel(
                            position=i,
                            direction=SeparatorModel.HORIZONTAL_SEPARATOR,
                            roi=block.get_bounds()
                        )
                        # 2‑pixel thickness; adjust if you want thicker visualized lines
                        separator.bounds = (block.get_x(), block.get_y() + i, width, 2)
                    else:
                        separator = SeparatorModel(
                            position=i,
                            direction=SeparatorModel.VERTICAL_SEPARATOR,
                            roi=block.get_bounds()
                        )
                        separator.bounds = (block.get_x() + i, block.get_y(), 2, height)
                    verified.append(separator)
        return verified
