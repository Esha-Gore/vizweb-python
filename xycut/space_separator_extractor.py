import numpy as np
import cv2
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block

class SpaceSeparatorExtractor:
    # Initializes the extractor with debug flag and std-dev threshold
    def __init__(self, debug=False, std_threshold=10):
        self.debug = debug
        self.std_threshold = std_threshold

    # Extracts whitespace-based separators using low-variance rows and columns
    def extract(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
        separators = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        row_std = np.std(gray, axis=1)
        col_std = np.std(gray, axis=0)

        horizontal_mask = row_std < self.std_threshold
        vertical_mask = col_std < self.std_threshold

        horizontal_segments = self._segment_boolean_mask(horizontal_mask)
        vertical_segments = self._segment_boolean_mask(vertical_mask)

        x, y, w, h = block.get_bounds()

        for seg in horizontal_segments:
            sy = y + seg['start']
            sh = seg['length']
            separator = SeparatorModel(
                position=sy,
                direction=SeparatorModel.HORIZONTAL_SEPARATOR,
                roi=(x, y, w, h)
            )
            separator.bounds = (x, sy, w, sh)
            separators.append(separator)

        for seg in vertical_segments:
            sx = x + seg['start']
            sw = seg['length']
            separator = SeparatorModel(
                position=sx,
                direction=SeparatorModel.VERTICAL_SEPARATOR,
                roi=(x, y, w, h)
            )
            separator.bounds = (sx, y, sw, h)
            separators.append(separator)

        return separators

    # Segments consecutive True values into ranges with start and length
    def _segment_boolean_mask(self, mask: np.ndarray) -> List[dict]:
        segments = []
        current_start = None

        for i, val in enumerate(mask):
            if val and current_start is None:
                current_start = i
            elif not val and current_start is not None:
                segments.append({'start': current_start, 'length': i - current_start})
                current_start = None

        if current_start is not None:
            segments.append({'start': current_start, 'length': len(mask) - current_start})

        return segments
