import cv2
import numpy as np
from xycut.separator_model import SeparatorModel
from typing import List
from xycut.block import Block
from typing import List


class SeparatorExtractor:
    def extract(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> List[SeparatorModel]:
        if image is None or image.size == 0:
            print("Warning: empty image passed to SeparatorExtractor.extract()")
            return []

        # --- Your debug print (kept) ---
        print(f"Running extract on block {block.bounds}")

        # todo: visualize image here
        # I used cv2.imread, image is BGR.
        # maybe BGR instead of RBG
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"Grayscale mean: {np.mean(gray):.2f}")
        # print grayscale
        # todo: visualze grayscale image here
        # todo: compare this section with java to confirm the meanings

        x, y, w, h = block.bounds
        bounds_str = f"{x}_{y}_{w}_{h}"
        cv2.imwrite(f"debug_output/gray_block_{bounds_str}.png", gray)

        # Canny thresholds to MATCH Java (0.66*50 ≈ 33, 1.33*50 ≈ 67) 
        # Aperture=3 matches Java's Sobel size.
        edges = cv2.Canny(gray, 33, 67, apertureSize=3)
        cv2.imwrite(f"debug_output/edges_block_{bounds_str}.png", edges)

        # todo: conform output of the edges, given images in gfg

        # Morphological closing (dilate then erode) to bridge tiny gaps (added interations no actual change)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        print(f"Closed nonzero: {np.count_nonzero(closed)}")
        # todo: confirm outout of closed, use 
        cv2.imwrite(f"debug_output/closed_block_{bounds_str}.png", closed)

        # Whitespace band detection 
        height, width = closed.shape
        threshold_h = width * 0.15
        threshold_v = height * 0.15

        row_sums = np.sum(closed < 10, axis=1)
        col_sums = np.sum(closed < 10, axis=0)

        print(f"Max row sum: {np.max(row_sums)} / Threshold_h: {threshold_h}")
        print(f"Max col sum: {np.max(col_sums)} / Threshold_v: {threshold_v}")

        separators: List[SeparatorModel] = []

        # Updated whitespace band detection
        min_band_thickness = 10
        min_empty_ratio = 0.95

        # Horizontal whitespace bands 
        for yy in range(0, height - min_band_thickness):
            band = closed[yy:yy + min_band_thickness, :]
            empty_ratio = np.mean(band < 10)
            if empty_ratio >= min_empty_ratio:
                sep = SeparatorModel(yy, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                sep.bounds = (block.get_x(), block.get_y() + yy, width, min_band_thickness)
                # print(f"Band sep (H): {sep.bounds} → thickness={sep.get_thickness()}, length={sep.get_length()}")
                separators.append(sep)

        # Vertical whitespace bands 
        for xx in range(0, width - min_band_thickness):
            band = closed[:, xx:xx + min_band_thickness]
            empty_ratio = np.mean(band < 10)
            if empty_ratio >= min_empty_ratio:
                sep = SeparatorModel(xx, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                sep.bounds = (block.get_x() + xx, block.get_y(), min_band_thickness, height)
                # print(f"Band sep (V): {sep.bounds} → thickness={sep.get_thickness()}, length={sep.get_length()}")
                separators.append(sep)

        # --- Optional line separator fallback ---
        if use_line_separators:
            print("Calling extract_line_separators")
            line_separators = self.extract_line_separators(image, block)
            print(f"Line separators found: {len(line_separators)}")
            separators.extend(line_separators)

        return separators

    def extract_line_separators(self, image: np.ndarray, block: Block) -> List[SeparatorModel]:
        separators: List[SeparatorModel] = []

        # This path is a binary‑morph fallback (not Canny‑based); kept as‑is and clarified.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple threshold to get strong lines; tweak 200 if UI is light/dark inverted.
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bx, by, _, _ = block.get_bounds()

        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 1 and w >= 30:
                sep = SeparatorModel(by + y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                sep.bounds = (bx + x, by + y, w, h)
                separators.append(sep)

        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 1 and h >= 30:
                sep = SeparatorModel(bx + x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                sep.bounds = (bx + x, by + y, w, h)
                separators.append(sep)

        return separators

    def extract_with_closed(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> tuple[List[SeparatorModel], np.ndarray]:
        if image is None or image.size == 0:
            print("Warning: empty image passed to extract_with_closed()")
            return [], np.zeros_like(image)

        print("in extract_with_closed")

        # Keep the same BGR→GRAY rule here too.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Java parity thresholds here as well (consistency across code paths) ---
        edges = cv2.Canny(gray, 33, 67, apertureSize=3)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # From here down, same separator logic; comments kept concise.
        separators: List[SeparatorModel] = []

        height, width = image.shape[:2]
        threshold_h = width * 0.15
        threshold_v = height * 0.15

        row_sums = np.sum(closed < 10, axis=1)
        col_sums = np.sum(closed < 10, axis=0)

        # Horizontal whitespace bands (banded segmenter)
        in_band = False
        start_y = 0
        for yy in range(height):
            if row_sums[yy] > threshold_h:
                if not in_band:
                    start_y = yy
                    in_band = True
            else:
                if in_band:
                    bh = yy - start_y
                    if bh >= 2:
                        sep = SeparatorModel(start_y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                        sep.bounds = (block.get_x(), block.get_y() + start_y, width, bh)
                        if sep.get_thickness() >= 3 and sep.get_length() >= 50:
                            print(f"Horizontal separator: {sep.bounds}")
                            separators.append(sep)
                    in_band = False

        # Vertical whitespace bands (banded segmenter)
        in_band = False
        start_x = 0
        for xx in range(width):
            if col_sums[xx] > threshold_v:
                if not in_band:
                    start_x = xx
                    in_band = True
            else:
                if in_band:
                    bw = xx - start_x
                    if bw >= 2:
                        sep = SeparatorModel(start_x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                        sep.bounds = (block.get_x() + start_x, block.get_y(), bw, height)
                        if sep.get_thickness() >= 3 and sep.get_length() >= 50:
                            print(f"Vertical separator: {sep.bounds}")
                            separators.append(sep)
                    in_band = False

        # Optional line separator fallback (binary‑morph path)
        if use_line_separators:
            line_separators = self.extract_line_separators(image, block)
            separators.extend(line_separators)

        return separators, closed




