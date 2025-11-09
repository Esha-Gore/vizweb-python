import numpy as np
import cv2
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block

class SpaceSeparatorExtractor:
    
    def __init__(self, debug: bool = True, base_std_threshold: int = 10):
        self.debug = debug
        self.base_std_threshold = base_std_threshold

    def extract(self, block: Block, image: np.ndarray) -> List[SeparatorModel]:
        bx, by, bw, bh = block.get_bounds()
        if bw <= 0 or bh <= 0:
            return []
        
        # Extract ROI
        roi_bgr = image[by:by+bh, bx:bx+bw]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Light blur to reduce noise, maybe will help idk trying anything. 
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Calculate standard deviation for each row and column
        row_std = np.std(gray, axis=1)  # stddev across each row
        col_std = np.std(gray, axis=0)  # stddev across each column
        
        # PRE-CHECK: Skip if no low-variation areas exist
        if col_std.min() > 25 and row_std.min() > 25:
            if self.debug:
                print(f"  SpaceSep: No low-variation areas (min col={col_std.min():.1f}, row={row_std.min():.1f})")
            return []
        
        # Use adaptive threshold based on the data, THRESHOLD like what we discussed
        col_threshold = min(max(self.base_std_threshold, np.percentile(col_std, 20)), 30)
        row_threshold = min(max(self.base_std_threshold, np.percentile(row_std, 20)), 30)
        
        if self.debug:
            print(f"  SpaceSep: Thresholds - col={col_threshold:.1f}, row={row_threshold:.1f}")
        
        # Create masks for "empty" rows/columns
        horizontal_mask = row_std < row_threshold
        vertical_mask = col_std < col_threshold
        
        if self.debug:
            print(f"  SpaceSep: Empty lines - {np.sum(horizontal_mask)} horiz, {np.sum(vertical_mask)} vert")
        
        # Find contiguous segments
        horizontal_segments = self._segment_boolean_mask(horizontal_mask)
        vertical_segments = self._segment_boolean_mask(vertical_mask)
        
        # Minimum thickness: 2% of dimension or 3 pixels (prevents 1-pixel noise)
        min_h_thickness = max(3, int(0.02 * bh))
        min_v_thickness = max(3, int(0.02 * bw))
        
        separators = []
        
        # Create horizontal separators
        for seg in horizontal_segments:
            sh = seg['length']
            
            # Filter by minimum thickness
            if sh < min_h_thickness:
                continue
            sy_rel = seg['start']
            band = gray[sy_rel:sy_rel+sh, :]
            white_ratio = np.sum(band > 200) / band.size
            
            if white_ratio < 0.85:
                if self.debug:
                    print(f"✗ Rejected H-sep: {white_ratio:.1%} white")
                continue
                
            sy = by + seg['start']
            separator = SeparatorModel(
                position=seg['start'],
                direction=SeparatorModel.HORIZONTAL_SEPARATOR,
                roi=(bx, by, bw, bh)
            )
            separator.bounds = (bx, sy, bw, sh)
            separator.source = "space_band_H"
            separators.append(separator)
        
        # Create vertical separators
        for seg in vertical_segments:
            sw = seg['length']
            
            # Filter by minimum thickness
            if sw < min_v_thickness:
                continue

            sx_rel = seg['start']
            band = gray[:, sx_rel:sx_rel+sw]
            white_ratio = np.sum(band > 200) / band.size
            
            if white_ratio < 0.85:
                if self.debug:
                    print(f"✗ Rejected V-sep: {white_ratio:.1%} white")
                continue
                
            sx = bx + seg['start']
            separator = SeparatorModel(
                position=seg['start'],
                direction=SeparatorModel.VERTICAL_SEPARATOR,
                roi=(bx, by, bw, bh)
            )
            separator.bounds = (sx, by, sw, bh)
            separator.source = "space_band_V"
            separators.append(separator)
        
        if self.debug and separators:
            print(f"  SpaceSep: Found {len(separators)} separators after filtering")
        
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
