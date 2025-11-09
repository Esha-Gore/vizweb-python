import numpy as np
import cv2
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block

class LineSeparatorExtractor:  
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.line_pixel_ratio_threshold = 0.85
        self.ratio_decrement = 0.1
        self.max_retries = 3
        
        # Neighbor verification threshold
        self.neighbor_threshold_ratio = 0.2

    def extract(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
        separators: List[SeparatorModel] = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cvCanny(gray, mask, 0.66*50, 1.33*50, 3) -> (33, 67, aperture=3).
        edges = cv2.Canny(gray, 33, 67, apertureSize=3)

        # Morphological closing to bridge tiny gaps between line pixels.
        # iterations=1 matches Java's single dilate + single erode.
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        height, width = edges.shape

        # Try with decreasing thresholds
        for retry in range(self.max_retries):
            threshold_h = width * (self.line_pixel_ratio_threshold - retry * self.ratio_decrement)
            threshold_v = height * (self.line_pixel_ratio_threshold - retry * self.ratio_decrement)

            row_counts = np.count_nonzero(edges, axis=1)
            col_counts = np.count_nonzero(edges, axis=0)

            horizontal = self._find_verified_separators(
                row_counts, threshold_h, 
                is_horizontal=True, 
                block=block, 
                width=width, 
                height=height
            )
            vertical = self._find_verified_separators(
                col_counts, threshold_v, 
                is_horizontal=False, 
                block=block, 
                width=width, 
                height=height
            )

            separators = horizontal + vertical
            
            if self.debug:
                print(f"  LineSep retry={retry}: Found {len(horizontal)} horiz, {len(vertical)} vert")
            
            if separators:
                break

        return separators

    def _find_verified_separators( self, values: np.ndarray, threshold: float, is_horizontal: bool, block: Block, width: int, height: int) -> List[SeparatorModel]:
        # Find positions where edge count exceeds threshold
        mask = np.array(values) > threshold
        segments = self._segment_boolean_mask(mask)

        verified = []
        for seg in segments:
            pos = seg['start']
            size = seg['length']

            # Skip full-height/full-width separators (usually false positives)
            if is_horizontal and size >= 0.98 * height:
                if self.debug:
                    print(f"    Rejected full-height separator at pos={pos}")
                continue
            if not is_horizontal and size >= 0.98 * width:
                if self.debug:
                    print(f"    Rejected full-width separator at pos={pos}")
                continue

            # This filters out text that looks like lines.
            
            before_pos = max(0, pos - 1)
            current_pos = pos
            after_pos = min(len(values) - 1, pos + size)
            
            before_response = values[before_pos]
            current_response = values[current_pos]
            after_response = values[after_pos]
            
            # Neighbors must have < 20% of the line's edge density
            is_low_before = before_response < self.neighbor_threshold_ratio * current_response
            is_low_after = after_response < self.neighbor_threshold_ratio * current_response
            
            if not (is_low_before and is_low_after):
                if self.debug:
                    print(f"    Rejected separator at pos={pos}: Failed neighbor verification")
                    print(f"      before={before_response:.1f}, current={current_response:.1f}, after={after_response:.1f}")
                    print(f"      before_ratio={before_response/current_response:.2f}, after_ratio={after_response/current_response:.2f}")
                continue
            
            if is_horizontal:
                separator = SeparatorModel(
                    position=pos,
                    direction=SeparatorModel.HORIZONTAL_SEPARATOR,
                    roi=block.get_bounds()
                )
                separator.bounds = (block.get_x(), block.get_y() + pos, width, size)
            else:
                separator = SeparatorModel(
                    position=pos,
                    direction=SeparatorModel.VERTICAL_SEPARATOR,
                    roi=block.get_bounds()
                )
                separator.bounds = (block.get_x() + pos, block.get_y(), size, height)

            verified.append(separator)
            
            if self.debug:
                print(f"    ✓ Verified {'H' if is_horizontal else 'V'} separator at pos={pos}, size={size}")

        return verified

    @staticmethod
    def _segment_boolean_mask(mask: np.ndarray) -> List[dict]:
        segments = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append({'start': start, 'length': i - start})
                start = None
        if start is not None:
            segments.append({'start': start, 'length': len(mask) - start})
        return segments