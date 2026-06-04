import numpy as np
import cv2
from typing import List
from xycut.separator_model import SeparatorModel
from xycut.block import Block

class SpaceSeparatorExtractor:
    
    def __init__(self, debug: bool = True, base_std_threshold: int = 10):
        self.debug = debug
        self.base_std_threshold = max(base_std_threshold, 25)

    def extract(self, block: Block, image: np.ndarray, h_min_thickness = 15, v_min_thickness = 15, r_threshold = 35, c_threshold = 30) -> List[SeparatorModel]:
        bx, by, bw, bh = block.get_bounds()
        if bw <= 0 or bh <= 0:
            return []
        
        # Extract ROI
        roi_bgr = image[by:by+bh, bx:bx+bw]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Light blur to reduce noise, maybe will help idk trying anything. 
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Calculate standard deviation for each row and column
        row_std = np.std(gray, axis=1)  # stddev across each row
        col_std = np.std(gray, axis=0)  # stddev across each column

        
        # PRE-CHECK: Skip if no low-variation areas exist
        if col_std.min() > 25 and row_std.min() > 25:
            if self.debug:
                print(f"  SpaceSep: No low-variation areas (min col={col_std.min():.1f}, row={row_std.min():.1f})")
            return []
        
        # Use adaptive threshold based on the data, THRESHOLD like what we discussed
        col_threshold = min(max(self.base_std_threshold, np.percentile(col_std, 20)), c_threshold)
        row_threshold = min(max(self.base_std_threshold, np.percentile(row_std, 20)), r_threshold)

        row_threshold = max(row_threshold, 25)
        col_threshold = max(col_threshold, 25)

        horizontal_mask = row_std < row_threshold
        vertical_mask = col_std < col_threshold

        #print(h_min_thickness)

        #print(f"  SpaceSep block ({bx},{by},{bw},{bh}): row_std range [{row_std.min():.1f}, {row_std.max():.1f}]")
        #print(f"  SpaceSep: Found {np.sum(row_std < row_threshold)} rows with std < {row_threshold:.1f}")
        
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
        min_h_thickness = max(h_min_thickness, int(0.02 * bh))
        min_v_thickness = max(v_min_thickness, int(0.02 * bw))
        
        separators = []
        
        # Create horizontal separators
        for seg in horizontal_segments:
            sh = seg['length']
            
            # Filter by minimum thickness
            if sh < min_h_thickness:
                continue
            
            # sy_local = seg['start']
    
            # # Check 10 pixels before the separator
            # if sy_local >= 10:
            #     before_region = gray[sy_local-10:sy_local, :]
            #     before_std = np.std(before_region)
            # else:
            #     before_std = 0
            
            # # Check 10 pixels after the separator
            # if sy_local + sh + 10 <= gray.shape[0]:
            #     after_region = gray[sy_local+sh:sy_local+sh+10, :]
            #     after_std = np.std(after_region)
            # else:
            #     after_std = 0
            
            # # Only accept if BOTH sides have high variation (text exists on both sides)
            # if before_std < 30 or after_std < 30:
            #     if self.debug:
            #         print(f"  Rejected H-sep at {sy_local}: before_std={before_std:.1f}, after_std={after_std:.1f}")
            #     continue

            aspect_ratio = bw / sh  # width / height
            if aspect_ratio < 10:  # Separator must be at least 10x wider than tall
                if self.debug:
                    print(f"  Rejected H-sep: aspect_ratio={aspect_ratio:.1f} < 10")
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

            aspect_ratio = bh / sw  # height / width
            if aspect_ratio < 10:  # Separator must be at least 10x taller than wide
                if self.debug:
                    print(f"  Rejected V-sep: aspect_ratio={aspect_ratio:.1f} < 10")
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