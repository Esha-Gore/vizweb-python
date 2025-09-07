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

        # --- 0) Prefer line separators (like Java) ---
        if use_line_separators:
            line_seps = self.extract_line_separators(image, block)
            if line_seps:
                return line_seps  # EARLY RETURN: lines win

        # --- 1) Fallback: whitespace bands on the ROI (Canny + close) ---
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

        # --- Work on the ROI ONLY ---
        roi_bgr = image[by:by+bh, bx:bx+bw]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # --- Robust binarization with Otsu both ways; pick the one with fewer whites ---
        _, bin_inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, bin_dir = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)

        def pick(binary_a, binary_b):
            a_ratio = float(np.mean(binary_a == 255))
            b_ratio = float(np.mean(binary_b == 255))
            # prefer fewer whites (foreground should be sparse)
            return binary_a if a_ratio <= b_ratio else binary_b

        binary = pick(bin_inv, bin_dir)

        # --- Kernels scale with ROI size (Java-like behavior but adaptive) ---
        hk = max(7, bw // 50)  # horizontal span
        vk = max(7, bh // 50)  # vertical span
        hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_k)
        vertical   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_k)

        contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(vertical,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Geometry guards (long & thin), and reject full-ROI blobs ---
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

    # def extract(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> List[SeparatorModel]:
    #     if image is None or image.size == 0:
    #         return []

    #     bx, by, bw, bh = block.get_bounds()
    #     gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     region    = gray_full[by:by+bh, bx:bx+bw]          # local

    #     edges  = cv2.Canny(region, 33, 67, apertureSize=3)
    #     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    #     height, width = closed.shape
    #     # row_sums = np.sum(closed < 10, axis=1)
    #     # col_sums = np.sum(closed < 10, axis=0)

    #     # # use thresholds relative to the BLOCK, not the full image
    #     # threshold_h = width  * 0.15
    #     # threshold_v = height * 0.15

    #     separators: List[SeparatorModel] = []

    #     if use_line_separators:
    #         line_seps = self.extract_line_separators(image, block)
    #         if line_seps:
    #             for s in line_seps:
    #                 s.source = "line"          # optional: for debugging
    #             return line_seps               # early return: lines win

    #     min_band_thickness = 10
    #     min_empty_ratio    = 0.95

    #     # --- Whitespace band detection (coalesced, not sliding windows) ---
    #     min_band_thickness = 10       # same as before
    #     min_empty_ratio    = 0.95     # same as before

    #     height, width = closed.shape
    #     bx, by, bw, bh = block.get_bounds()

    #     # Boolean masks for "empty enough" rows/columns
    #     row_empty = (np.mean(closed < 10, axis=1) >= min_empty_ratio)
    #     col_empty = (np.mean(closed < 10, axis=0) >= min_empty_ratio)

    #     # 1) Horizontal whitespace bands (contiguous runs of empty rows)
    #     y = 0
    #     while y < height:
    #         if row_empty[y]:
    #             start = y
    #             while y < height and row_empty[y]:
    #                 y += 1
    #             band_h = y - start
    #             if band_h >= min_band_thickness:
    #                 sep = SeparatorModel(start, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
    #                 sep.bounds = (bx, by + start, width, band_h)  # full block width across the band
    #                 sep.source = "space_band_H"
    #                 separators.append(sep)
    #         else:
    #             y += 1

    #     # 2) Vertical whitespace bands (contiguous runs of empty columns)
    #     x = 0
    #     while x < width:
    #         if col_empty[x]:
    #             start = x
    #             while x < width and col_empty[x]:
    #                 x += 1
    #             band_w = x - start
    #             if band_w >= min_band_thickness:
    #                 sep = SeparatorModel(start, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
    #                 sep.bounds = (bx + start, by, band_w, height)  # full block height across the band
    #                 sep.source = "space_band_V"
    #                 separators.append(sep)
    #         else:
    #             x += 1


    #     if use_line_separators:
    #         separators.extend(self.extract_line_separators(image, block))

    #     # sanity check — catches missed offsets immediately
    #     H, W = image.shape[:2]
    #     for s in separators:
    #         x,y,w,h = s.get_bounds()
    #         assert w > 0 and h > 0 and 0 <= x < x+w <= W and 0 <= y < y+h <= H
            

    #     return separators


    # def extract_line_separators(self, image: np.ndarray, block: Block) -> List[SeparatorModel]:
    #     separators: List[SeparatorModel] = []
    #     bx, by, bw, bh = block.get_bounds()

    #     gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    #     # work on local region so contour coords are LOCAL
    #     region = binary[by:by+bh, bx:bx+bw]

    #     horizontal = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    #     vertical   = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

    #     contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours_v, _ = cv2.findContours(vertical,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     for cnt in contours_h:
    #         x, y, w, h = cv2.boundingRect(cnt)  # LOCAL to region
    #         if h >= 1 and w >= 30:
    #             sep = SeparatorModel(by + y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
    #             sep.bounds = (bx + x, by + y, w, h)  # local→global
    #             separators.append(sep)

    #     for cnt in contours_v:
    #         x, y, w, h = cv2.boundingRect(cnt)      # LOCAL to region
    #         if w >= 1 and h >= 30:
    #             sep = SeparatorModel(bx + x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
    #             sep.bounds = (bx + x, by + y, w, h)  # local→global
    #             separators.append(sep)

    #     H, W = image.shape[:2]
    #     for s in separators:
    #         x, y, w, h = s.get_bounds()
    #         assert w > 0 and h > 0 and 0 <= x < x + w <= W and 0 <= y < y + h <= H, \
    #             f"Non-global or OOB separator: {s.get_bounds()} for image {W}x{H}"

    #     return separators


    # def extract_with_closed(self, block: Block, image: np.ndarray, use_line_separators: bool = True) -> tuple[List[SeparatorModel], np.ndarray]:
    #     if image is None or image.size == 0:
    #         return [], np.zeros_like(image)

    #     bx, by, bw, bh = block.get_bounds()
    #     gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     region    = gray_full[by:by+bh, bx:bx+bw]

    #     edges  = cv2.Canny(region, 33, 67, apertureSize=3)
    #     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    #     height, width = region.shape
    #     row_sums = np.sum(closed < 10, axis=1)
    #     col_sums = np.sum(closed < 10, axis=0)
    #     threshold_h = width  * 0.15
    #     threshold_v = height * 0.15

    #     separators: List[SeparatorModel] = []

    #     # Horizontal banded
    #     in_band, start_y = False, 0
    #     for yy in range(height):
    #         if row_sums[yy] > threshold_h:
    #             if not in_band: start_y, in_band = yy, True
    #         else:
    #             if in_band:
    #                 bh_loc = yy - start_y
    #                 if bh_loc >= 2:
    #                     sep = SeparatorModel(start_y, SeparatorModel.HORIZONTAL_SEPARATOR, block.get_bounds())
    #                     sep.bounds = (bx, by + start_y, bw, bh_loc)  # local→global
    #                     if sep.get_thickness() >= 3 and sep.get_length() >= 50:
    #                         separators.append(sep)
    #                 in_band = False

    #     # Vertical banded
    #     in_band, start_x = False, 0
    #     for xx in range(width):
    #         if col_sums[xx] > threshold_v:
    #             if not in_band: start_x, in_band = xx, True
    #         else:
    #             if in_band:
    #                 bw_loc = xx - start_x
    #                 if bw_loc >= 2:
    #                     sep = SeparatorModel(start_x, SeparatorModel.VERTICAL_SEPARATOR, block.get_bounds())
    #                     sep.bounds = (bx + start_x, by, bw_loc, bh)  # local→global
    #                     if sep.get_thickness() >= 3 and sep.get_length() >= 50:
    #                         separators.append(sep)
    #                 in_band = False

    #     if use_line_separators:
    #         separators.extend(self.extract_line_separators(image, block))

    #     H, W = image.shape[:2]
    #     for s in separators:
    #         x, y, w, h = s.get_bounds()
    #         assert w > 0 and h > 0 and 0 <= x < x + w <= W and 0 <= y < y + h <= H, \
    #             f"Non-global or OOB separator: {s.get_bounds()} for image {W}x{H}"


    #     # optional: return the local closed image for debugging (it’s local!)
    #     return separators, closed





