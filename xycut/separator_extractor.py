import cv2
import numpy as np
from xycut.separator_model import SeparatorModel
from typing import List
from xycut.block import Block
from typing import List


class SeparatorExtractor:

    def extract(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> List[SeparatorModel]:
        if image is None or image.size == 0:
            return []

        bx, by, bw, bh = block.get_bounds()
        if bw <= 0 or bh <= 0:
            return []

        #  Prefer line separators 
        if use_line_separators:
            line_seps = self.extract_line_separators(image, block)
            if line_seps:
                return line_seps  # lines win

        roi_bgr = image[by:by+bh, bx:bx+bw]

        gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 33, 67, apertureSize=3)
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        h, w = closed.shape
        sep_list: List[SeparatorModel] = []

        min_band_thickness = 10
        min_empty_ratio    = 0.95

        row_empty = (np.mean(closed < 10, axis=1) >= min_empty_ratio)
        col_empty = (np.mean(closed < 10, axis=0) >= min_empty_ratio)

        # Horizontal runs in ROI coords
        y = 0
        while y < h:
            if row_empty[y]:
                start = y
                while y < h and row_empty[y]:
                    y += 1
                band_h = y - start
                if band_h >= min_band_thickness:
                    sep = SeparatorModel(start, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                    sep.bounds = (bx, by + start, w, band_h)
                    sep.source = "space_band_H"
                    sep_list.append(sep)
            else:
                y += 1

        # Vertical runs in ROI coords
        x = 0
        while x < w:
            if col_empty[x]:
                start = x
                while x < w and col_empty[x]:
                    x += 1
                band_w = x - start
                if band_w >= min_band_thickness:
                    sep = SeparatorModel(start, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                    sep.bounds = (bx + start, by, band_w, h)
                    sep.source = "space_band_V"
                    sep_list.append(sep)
            else:
                x += 1

        return sep_list


    def extract_line_separators(self, image: np.ndarray, block: Block) -> List[SeparatorModel]:
        separators: List[SeparatorModel] = []

        bx, by, bw, bh = block.get_bounds()
        if bw <= 0 or bh <= 0:
            return separators

        roi_bgr = image[by:by+bh, bx:bx+bw]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        _, bin_inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, bin_dir = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

        def pick(binary_a, binary_b):
            a_ratio = float(np.mean(binary_a == 255))
            b_ratio = float(np.mean(binary_b == 255))
            # prefer fewer whites 
            return binary_a if a_ratio <= b_ratio else binary_b

        binary = pick(bin_inv, bin_dir)

        hk = max(7, bw // 50)  # horizontal span
        vk = max(7, bh // 50)  # vertical span
        hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_k)
        vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_k)

        contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(vertical,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_len_h   = max(int(0.25 * bw), 50)
        max_thick_h = max(1, int(0.02 * bh))
        min_len_v   = max(int(0.25 * bh), 50)
        max_thick_v = max(1, int(0.02 * bw))

        for cnt in contours_h:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= max_thick_h and w >= min_len_h and not (w >= 0.98 * bw and h >= 0.98 * bh):
                sep = SeparatorModel(y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
                sep.bounds = (bx + x, by + y, w, h)
                sep.source = "line"
                separators.append(sep)

        for cnt in contours_v:
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= max_thick_v and h >= min_len_v and not (w >= 0.98 * bw and h >= 0.98 * bh):
                sep = SeparatorModel(x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
                sep.bounds = (bx + x, by + y, w, h)
                sep.source = "line"
                separators.append(sep)

        return separators





