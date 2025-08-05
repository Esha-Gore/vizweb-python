import cv2
import numpy as np
from xycut.separator_model import SeparatorModel
from typing import List
from xycut.block import Block

class SeparatorExtractor:
    def extract(self, block: Block, image: np.ndarray, use_line_separators=True) -> List[SeparatorModel]:
        if image is None or image.size == 0:
            print("Warning: empty image passed to SeparatorExtractor.extract()")
            return []

        separators = []
        print("in extract")

        print(f"Input image shape: {image.shape}")

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(f"Grayscale mean intensity: {np.mean(gray)}")

        edges = cv2.Canny(gray, 50, 150)
        print(f"Edge map nonzero pixels: {np.count_nonzero(edges)}")

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        print(f"Closed map nonzero pixels: {np.count_nonzero(closed)}")

        height, width = image.shape[:2]
        threshold_h = width * 0.15
        threshold_v = height * 0.15

        row_sums = np.sum(closed < 10, axis=1)
        col_sums = np.sum(closed < 10, axis=0)

        print(f"Max row sum: {np.max(row_sums)} / threshold_h: {threshold_h}")
        print(f"Max col sum: {np.max(col_sums)} / threshold_v: {threshold_v}")
        print(f"Rows above threshold: {np.sum(row_sums > threshold_h)}")
        print(f"Cols above threshold: {np.sum(col_sums > threshold_v)}")

        # --- Horizontal whitespace bands ---
        in_band = False
        start_y = 0

        for y in range(height):
            if row_sums[y] > threshold_h:
                if not in_band:
                    start_y = y
                    in_band = True
            else:
                if in_band:
                    band_height = y - start_y
                    if band_height >= 2:
                        sep = SeparatorModel(start_y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                        sep.bounds = (block.get_x(), block.get_y() + start_y, width, band_height)
                        if sep.get_thickness() >= 3 and sep.get_length() >= 50:
                            print(f"Horizontal separator: {sep.bounds}")
                            separators.append(sep)
                    in_band = False

        # --- Vertical whitespace bands ---
        in_band = False
        start_x = 0

        for x in range(width):
            if col_sums[x] > threshold_v:
                if not in_band:
                    start_x = x
                    in_band = True
            else:
                if in_band:
                    band_width = x - start_x
                    if band_width >= 2:
                        sep = SeparatorModel(start_x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                        sep.bounds = (block.get_x() + start_x, block.get_y(), band_width, height)
                        if sep.get_thickness() >= 3 and sep.get_length() >= 50:
                            print(f"Vertical separator: {sep.bounds}")
                            separators.append(sep)
                    in_band = False

        # --- Optional line separator fallback ---
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

        bx, by, _, _ = block.get_bounds()

        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            sep = SeparatorModel(by + y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
            #print(f"Raw separator bounds: {sep.bounds}")
            # Skip separators that are too small
            if sep.get_thickness() < 3 or sep.get_length() < 50:
                continue

            sep.bounds = (bx + x, by + y, w, h)
            separators.append(sep)

        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            sep = SeparatorModel(bx + x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
            #print(f"Raw separator bounds: {sep.bounds}")
            # Skip separators that are too small
            if sep.get_thickness() < 3 or sep.get_length() < 50:
                continue
            sep.bounds = (bx + x, by + y, w, h)
            separators.append(sep)

        return separators


