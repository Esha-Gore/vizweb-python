from typing import List, Tuple
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.separator_extractor import SeparatorExtractor
from xycut.separator_model import SeparatorModel
from xycut.block import Block
import uuid
import cv2
import os
import numpy as np

from typing import List

class XYDecomposer:
    def __init__(self):
        self.separators_used: List[Tuple[int, int, int, int]] = []

    # Entry point: Decomposes the full image into a block tree.
    def decompose(self, image, strategy: DefaultXYDecompositionStrategy) -> Block:
        self.separators_used.clear()
        root = Block()
        height, width = image.shape[:2]
        root.set_bounds((0, 0, width, height))
        self._recursive_decompose(image, root, strategy, level=0)
        return root

    # Recursively splits the block using separators according to the strategy
    def _recursive_decompose(self, image, block: Block, strategy: DefaultXYDecompositionStrategy, level: int):
    

        # stop if block is too small or max depth is exceeded
        if self._should_stop(block, strategy, level, image):
            return

        # reemove borders/margins if strategy allows
        # todo: potentially remove
        # if strategy.remove_border:
        #     new_bounds = strategy.find_content_bounds(image, block.bounds)
        #     if new_bounds != block.bounds:
        #         block.set_bounds(new_bounds)

        # combine separators from multiple sources
        # problem somewhere here
        separators = []

        if strategy.use_line_separators:
            #print(f"[Level {level}] Extracting line separators for block {block.bounds}")    
            separators += strategy.line_separator_extractor.extract(block, image)

        if strategy.use_space_separators:
            #print(f"[Level {level}] Extracting space separators for block {block.bounds}")
            separators += strategy.space_separator_extractor.extract(block, image)

        print(f"[Level {level}] Extracted {len(separators)} separators")
        selected = strategy.choose_separators(separators)
        print(f"[Level {level}] Selected {len(selected)} separators")
        selected = strategy.choose_separators(separators)
        if not selected:
            return
        selected.sort(key=lambda sep: sep.get_y() if sep.is_horizontal() else sep.get_x())

        
        # visual debugging
        strategy.visualize_separators(image, block.bounds, selected, level)



        # for each selected separator, split block and recurse on each part
        for sep in selected:
            self.separators_used.append((sep.get_x(), sep.get_y(), sep.get_width(), sep.get_height()))
            print(f"Block bounds: {block.bounds}")
            print(f"Separator bounds: {sep.get_bounds()}")

            if sep.get_width() <= 0 or sep.get_height() <= 0:
                continue
            if sep.get_x() < 0 or sep.get_y() < 0 or sep.get_x() + sep.get_width() > image.shape[1] or sep.get_y() + sep.get_height() > image.shape[0]:
                continue

            bx, by, bw, bh = block.get_bounds()
            sx, sy, sw, sh = sep.get_bounds()

            # Allow edge separators as long as at least one valid sub-block is produced
            if sep.is_horizontal():
                top_h = sy - by
                bottom_h = (by + bh) - (sy + sh)
                if top_h <= 0 and bottom_h <= 0:
                    print(f"Skipping horizontal separator at y={sy} — no valid split (top_h={top_h}, bottom_h={bottom_h})")
                    continue
            else:
                left_w = sx - bx
                right_w = (bx + bw) - (sx + sw)
                if left_w <= 0 and right_w <= 0:
                    print(f"Skipping vertical separator at x={sx} — no valid split (left_w={left_w}, right_w={right_w})")
                    continue



            print(f"Attempting to split block: {block.get_bounds()} with separator: {sep.bounds}")

            sub_blocks = self._split_block(block, sep)

            print(f"Number of child blocks returned: {len(sub_blocks)}")

            for sb in sub_blocks:

                #print(f"[Level {level+1}] Sub-block bounds: {sb.bounds}")
                print(f"[Level {level}] → Child block bounds: {sb.bounds}")
                x, y, w, h = sb.bounds
                sub_image = image[y:y+h, x:x+w]
                #print(f"[Level {level}] → sub_image shape: {sub_image.shape}")
                
                if sub_image.size == 0:
                    #print(f"[Level {level}] → Skipping empty sub-image")
                    continue

                if self._should_stop(sb, strategy, level + 1, image):
                    #print(f"[Level {level+1}] → Stopping: block too small or too deep")
                    continue

                # Draw and save each sub_image with its level and ID
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.imwrite(f"debug/decomp_level{level}_{uuid.uuid4().hex[:6]}.png", debug_img)


                self._recursive_decompose(sub_image, sb, strategy, level + 1)


        # Splits a block into two based on a horizontal or vertical separator.
    def _split_block(self, block: Block, separator: SeparatorModel) -> List[Block]:
        x, y, w, h = block.bounds
        sep_x, sep_y, sep_w, sep_h = separator.get_bounds()
        blocks = []

        # Clip separator to block bounds
        sep_x = max(sep_x, x)
        sep_y = max(sep_y, y)
        sep_w = min(sep_w, x + w - sep_x)
        sep_h = min(sep_h, y + h - sep_y)

        if separator.is_horizontal():
            top_h = sep_y - y
            bottom_h = (y + h) - (sep_y + sep_h)

            if top_h > 0:
                top = Block()
                top.set_bounds((x, y, w, top_h))
                print(f"Created top block: {top.get_bounds()}")
                blocks.append(top)
            else:
                print(f"Top block too small (height={top_h}), skipping")

            if bottom_h > 0:
                bottom = Block()
                bottom.set_bounds((x, sep_y + sep_h, w, bottom_h))
                print(f"Created bottom block: {bottom.get_bounds()}")
                blocks.append(bottom)
            else:
                print(f"Bottom block too small (height={bottom_h}), skipping")

        else:  # vertical
            left_w = sep_x - x
            right_w = (x + w) - (sep_x + sep_w)

            if left_w > 0:
                left = Block()
                left.set_bounds((x, y, left_w, h))
                print(f"Created left block: {left.get_bounds()}")
                blocks.append(left)
            else:
                print(f"Left block too small (width={left_w}), skipping")

            if right_w > 0:
                right = Block()
                right.set_bounds((sep_x + sep_w, y, right_w, h))
                print(f"Created right block: {right.get_bounds()}")
                blocks.append(right)
            else:
                print(f"Right block too small (width={right_w}), skipping")

        return blocks


    def visualize_separators(self, image, bounds, separators, level):
        os.makedirs("xydebug", exist_ok=True)
        img_copy = image.copy()
        for sep in separators:
            x, y, w, h = sep.get_bounds()
            color = (0, 255, 0) if sep.is_horizontal() else (255, 0, 0)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        cv2.imwrite(f"xydebug/separators_level{level}.png", img_copy)


    #Checks whether to stop recursion based on block size and level.
    def _should_stop(self, block: Block, strategy: DefaultXYDecompositionStrategy, level: int, image) -> bool:
        x, y, w, h = block.bounds
        area = w * h
        region = image[y:y+h, x:x+w]
        # gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        # mean_intensity = np.mean(gray)

        # if mean_intensity > 240:
        #     print(f"[Level {level}] Region too white (mean={mean_intensity:.2f}), stopping recursion.")
        #     return True
        stop = (
            area < strategy.min_area or
            w < strategy.min_width or
            h < strategy.min_height or
            level > strategy.max_level
        )



        if stop:
            print(f"[Level {level}] STOP: Block {(x, y, w, h)} — area={area}, w={w}, h={h}")

        return stop

