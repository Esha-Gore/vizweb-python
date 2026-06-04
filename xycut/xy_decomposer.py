from typing import List, Tuple
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.separator_extractor import SeparatorExtractor
from xycut.separator_model import SeparatorModel
from xycut.block import Block
from xycut.space_analyzer import find_content_bounds
import uuid
import cv2
import os
import numpy as np

from typing import List

class XYDecomposer:
    def __init__(self, h_min_thickness: int = 15, v_min_thickness: int = 15, r_threshold: int = 20, c_threshold: int = 20):
        self.separators_used: List[Tuple[int, int, int, int]] = []
        self.h_min_thickness = h_min_thickness
        self.v_min_thickness = v_min_thickness
        self.r_threshold = r_threshold
        self.c_threshold = c_threshold

    # Entry point: Decomposes the full image into a block tree.
    def decompose(self, image, strategy: DefaultXYDecompositionStrategy) -> Block:
        strategy.use_space_separators = False
        self.separators_used.clear()
        root = Block()
        height, width = image.shape[:2]
        root.set_bounds((0, 0, width, height))
        self._recursive_decompose(image, root, strategy, level=0)
        return root

    def _recursive_decompose(self, image, block: Block, strategy: DefaultXYDecompositionStrategy, level: int):
        bx, by, bw, bh = block.get_bounds()

        # see if this makes it work
        if bw <= 0 or bh <= 0: 
            return

        # stop if block is too small or max depth is exceeded
        if self._should_stop(block, strategy, level, image):
            return
        
        # I don't think I remove the margins ----> doing now

        if strategy.remove_border:
            content_bounds = find_content_bounds(image, (bx, by, bw, bh))
            cx, cy, cw, ch = content_bounds
            
            # If content bounds are smaller than block bounds (there's margin/whitespace)
            if cw < bw or ch < bh:
                # Create inner block with trimmed bounds
                inner_block = Block()
                inner_block.set_bounds(content_bounds)
                
                # Recurse on the trimmer block
                self._recursive_decompose(image, inner_block, strategy, level)
                
                # Add the inner block as child
                block.add_child(inner_block)
                
                return

        # They may be doing something more complex to pick the extractors here
        #print(strategy.use_space_separators)
        extractors = []
        if strategy.use_line_separators and bw > 100 and bh > 100:
            extractors.append("line")
        if strategy.use_space_separators:
            extractors.append("space")

        selected = None

        # what does this part do? extract all the lines
        for kind in extractors:
            if kind == "line":
                cands = strategy.line_separator_extractor.extract(block, image)
                #self.dbg_list(cands, block, level, "line candidates")

                #cands = self._filter_bad_separators(cands, block, strategy)

                print(f"[DEBUG] Line extractor returned {len(cands)} separators for block ({bx},{by},{bw},{bh}):")
                for c in cands:
                    cx, cy, cw, ch = c.get_bounds()
                    print(f"  - at ({cx},{cy}) size ({cw},{ch})")
                
                            

                cands = [s for s in cands if s.get_length() > 100]
                if cands:
                    for s in cands:
                        s.source = "line"
                    selected = cands
                    break 

            else:  # "space"
                # some problem here. 
                cands = strategy.space_separator_extractor.extract(block, image)
                self.dbg_list(cands, block, level, "space candidates")

                #cands = self._filter_bad_separators(cands, block, strategy)
                
                selected = strategy.choose_separators(cands) if cands else []
                if selected:
                    for s in selected:
                        s.source = "space"
                    break


        # no separators 
        if not selected:
            return

        selected.sort(key=lambda s: (0 if s.is_horizontal() else 1,
                                    s.get_y() if s.is_horizontal() else s.get_x()))

        # (splitRectangleIntoRegions)
        regions = [block]
        for sep in selected:
            next_regions = []
            for r in regions:
                rx, ry, rw, rh = r.get_bounds()
                sx, sy, sw, sh = sep.get_bounds()

               
                if sw <= 0 or sh <= 0:
                    next_regions.append(r); continue
                # if no overlap with this region, keep it
                if sx + sw <= rx or sy + sh <= ry or sx >= rx + rw or sy >= ry + rh:
                    next_regions.append(r); continue

                parts = self._split_block(r, sep)  
                if not parts or len(parts) == 1:
                    next_regions.append(r)          # ineffective split, keep region
                else:
                    self.separators_used.append((sx, sy, sw, sh))  # sep worked
                    next_regions.extend(parts)

            regions = next_regions

        if selected:
            print(f"\n{'='*70}")
            print(f"USING {len(selected)} SEPARATORS for block ({bx},{by},{bw},{bh})")
            for i, sep in enumerate(selected):
                sx, sy, sw, sh = sep.get_bounds()
                source = getattr(sep, 'source', 'unknown')
                direction = 'H' if sep.is_horizontal() else 'V'
                print(f"  Sep {i+1}: {direction} source={source} at ({sx},{sy}) size ({sw},{sh})")
            print(f"{'='*70}\n")

        # add children and recurse once
        for child in regions:
            if child is block:
                continue
            block.add_child(child)
            self._recursive_decompose(image, child, strategy, level + 1)



        # Splits a block into two based on a horizontal or vertical separator.
    def _split_block(self, block: Block, separator: SeparatorModel) -> List[Block]:
        bx, by, bw, bh = block.get_bounds()
        sx, sy, sw, sh = separator.get_bounds()
        blocks: List[Block] = []

        # reject if outside block
        if sx + sw <= bx or sy + sh <= by or sx >= bx + bw or sy >= by + bh:
            return blocks

        # clip separator to block
        cx  = max(sx, bx)
        cy  = max(sy, by)
        cxe = min(sx + sw, bx + bw)
        cye = min(sy + sh, by + bh)
        cw, ch = cxe - cx, cye - cy
        if cw <= 0 or ch <= 0:
            return blocks

        if separator.is_horizontal():
            top_h    = cy - by
            bottom_h = (by + bh) - (cy + ch)
            if top_h > 0:
                top = Block(); top.set_bounds((bx, by, bw, top_h)); blocks.append(top)
            if bottom_h > 0:
                bot = Block(); bot.set_bounds((bx, cy + ch, bw, bottom_h)); blocks.append(bot)
        else:
            left_w  = cx - bx
            right_w = (bx + bw) - (cx + cw)
            if left_w > 0:
                left = Block(); left.set_bounds((bx, by, left_w, bh)); blocks.append(left)
            if right_w > 0:
                right = Block(); right.set_bounds((cx + cw, by, right_w, bh)); blocks.append(right)
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

        roi = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        white_ratio = np.sum(gray > 240) / gray.size
        
        if white_ratio > 0.95:  # More than 95% white
            return True

        # print(strategy.min_area)

        stop = (
            area < strategy.min_area or
            w < strategy.min_width or
            h < strategy.min_height or
            level > strategy.max_level
        )

        return stop


    def dbg_list(self, seps, block, level, label):
        bx, by, bw, bh = block.get_bounds()
        print(f"[L{level}] {label}: {len(seps)}")
        for i, s in enumerate(seps):
            x,y,w,h = s.get_bounds()
            length = w if s.is_horizontal() else h
            thick  = h if s.is_horizontal() else w
            lr = length / (bw if s.is_horizontal() else bh)
            tr = thick  / (bh if s.is_horizontal() else bw)
            print(f" #{i:02d} src={getattr(s,'source',None)} dir={'H' if s.is_horizontal() else 'V'} "
                f"b={x,y,w,h} len={length}({lr:.2f}) thick={thick}({tr:.3f})", flush=True)
            
    def _filter_bad_separators(self, separators: List[SeparatorModel], block: Block, strategy:DefaultXYDecompositionStrategy) -> List[SeparatorModel]:
        bx, by, bw, bh = block.get_bounds()
        MIN_WIDTH = 40 #strategy.min_width
        MIN_HEIGHT = 20  #strategy.min_height
        # MIN_THICKNESS =  15#strategy.min_thickness
        
        print(f"  Filtering {len(separators)} seps for block ({bx},{by},{bw},{bh})")
        
            
        filtered = []
        for sep in separators:
            sx, sy, sw, sh = sep.get_bounds()
            
            # # Reject very thin separators (likely borders/noise)
            # thickness = sh if sep.is_horizontal() else sw
            # if thickness < MIN_THICKNESS:
            #     continue
            
            if sep.is_horizontal():
                top_h = sy - by
                bottom_h = (by + bh) - (sy + sh)
                
                if top_h > 0 and top_h < MIN_HEIGHT:
                    print(f"    Rejected H-sep: top={top_h} < {MIN_HEIGHT}")
                    continue
                if bottom_h > 0 and bottom_h < MIN_HEIGHT:
                    print(f"    Rejected H-sep: bottom={bottom_h} < {MIN_HEIGHT}")
                    continue
            
            else:  # vertical
                left_w = sx - bx
                right_w = (bx + bw) - (sx + sw)
                
                if left_w > 0 and left_w < MIN_WIDTH:
                    print(f"    Rejected V-sep at x={sx}: left={left_w} < {MIN_WIDTH}")
                    continue
                if right_w > 0 and right_w < MIN_WIDTH:
                    print(f"    Rejected V-sep at x={sx}: right={right_w} < {MIN_WIDTH}")
                    continue
            
            filtered.append(sep)
        
        print(f"  → Kept {len(filtered)}/{len(separators)}")
        return filtered

