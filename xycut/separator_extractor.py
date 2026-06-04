import cv2
import numpy as np
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block
from xycut.line_separator_extractor import LineSeparatorExtractor
from xycut.space_separator_extractor import SpaceSeparatorExtractor


class SeparatorExtractor:
    
    def __init__(self, debug: bool = False):
        self.debug = debug

        self._line_extractor = LineSeparatorExtractor(debug=debug)
        self._space_extractor = SpaceSeparatorExtractor(debug=debug, base_std_threshold=10)
    
    def extract(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> List[SeparatorModel]:
        if image is None or image.size == 0:
            return []

        bx, by, bw, bh = block.get_bounds()
        if bw <= 0 or bh <= 0:
            return []

        # Try line separators first for large regions
        roi_large_enough = (bw >= 100 and bh >= 100)
        if use_line_separators and roi_large_enough:
            line_seps = self._line_extractor.extract(block, image)
            if line_seps:
                return line_seps

        # Fall back to space separators
        return self._space_extractor.extract(block, image)
    
    def extract_line_separators(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
        return self._line_extractor.extract(block, image)
    
    # def extract_space_separators(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
    #     return self._space_extractor.extract(block, image, h_min_thickness, v_min_thickness,r_threshold, c_threshold)    





