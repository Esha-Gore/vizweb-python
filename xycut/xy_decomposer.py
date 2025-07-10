from typing import List, Tuple
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.separator_extractor import SeparatorExtractor
from xycut.separator_model import SeparatorModel
from xycut.block import Block


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
        if self._should_stop(block, strategy, level):
            return

        # reemove borders/margins if strategy allows
        if strategy.remove_border:
            new_bounds = strategy.find_content_bounds(image, block.bounds)
            if new_bounds != block.bounds:
                block.set_bounds(new_bounds)

        # combine separators from multiple sources
        separators = []
        if strategy.use_line_separators:
            separators += strategy.line_separator_extractor.extract(block, image)
        if strategy.use_space_separators:
            separators += strategy.space_separator_extractor.extract(block, image)

        # select best separators using strategy filter
        selected = strategy.choose_separators(separators)
        if not selected:
            return
        
        # visual debugging
        strategy.visualize_separators(image, block.bounds, selected, level)


        # for each selected separator, split block and recurse on each part
        for sep in selected:
            self.separators_used.append((sep.get_x(), sep.get_y(), sep.get_width(), sep.get_height()))
            sub_blocks = self._split_block(block, sep)
            for sb in sub_blocks:
                block.add_child(sb)
                x, y, w, h = sb.bounds
                sub_image = image[y:y+h, x:x+w]
                self._recursive_decompose(sub_image, sb, strategy, level + 1)

    # Splits a block into two based on a horizontal or vertical separator.
    def _split_block(self, block: Block, separator: SeparatorModel) -> List[Block]:
        x, y, w, h = block.bounds
        sep_x, sep_y, sep_w, sep_h = separator.get_bounds()

        if separator.is_horizontal():
            # Split into top and bottom
            top = Block()
            top.set_bounds((x, y, w, sep_y - y))

            bottom = Block()
            bottom.set_bounds((x, sep_y + sep_h, w, y + h - (sep_y + sep_h)))
            return [top, bottom]
        else:
            # Split into left and right
            left = Block()
            left.set_bounds((x, y, sep_x - x, h))

            right = Block()
            right.set_bounds((sep_x + sep_w, y, x + w - (sep_x + sep_w), h))
            return [left, right]

    #Checks whether to stop recursion based on block size and level.
    def _should_stop(self, block: Block, strategy: DefaultXYDecompositionStrategy, level: int) -> bool:
        x, y, w, h = block.bounds
        area = w * h
        return (
            area < strategy.min_area or
            w < strategy.min_width or
            h < strategy.min_height or
            level > strategy.max_level
        )

