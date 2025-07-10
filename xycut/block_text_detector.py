import numpy as np
from xycut.block import Block
from xycut.block_type import BlockType
from quadtree.completed_quadtree_feature_computer import FullQuadtreeFeatureComputer

class BlockTextDetector:
    # Initializes the detector with a block and its image region
    def __init__(self, block: Block, image: np.ndarray):
        self.block = block
        self.image = image
        self.detected_positive = False

    # Determines if the block should be labeled as text based on structure and entropy
    def detect(self):
        if self.block.is_leaf():
            self.detected_positive = False
            return

        x, y, w, h = self.block.get_bounds()
        aspect_ratio = h / w if w != 0 else 0

        children = self.block.get_children()
        num_children = len(children)

        if aspect_ratio < 0.5 and self.is_reasonable_to_be_text(self.block):
            if num_children >= 5:
                self.detected_positive = True
            elif num_children >= 3:
                avg_spacing = self.get_average_spacing(children)
                self.detected_positive = avg_spacing <= 25
            elif num_children >= 2:
                all_children_okay = all(self.is_reasonable_to_be_text(c) for c in children)
                if all_children_okay:
                    grand_children = []
                    for c in children:
                        grand_children.extend(c.get_children())
                    if len(grand_children) > 1:
                        avg_spacing = self.get_average_spacing(grand_children)
                        self.detected_positive = avg_spacing <= 15

        # fallback: quadtree validation for large blocks
        if self.detected_positive and self.block.get_area() > 8000 and h > 50:
            cropped_image = self.image[y:y+h, x:x+w]
            qt_detector = FullQuadtreeFeatureComputer(
                max_depth=5,
                min_size=20,
                entropy_threshold=3.0
            )
            qt_detector.compute_features(cropped_image)
            if qt_detector.count_leaves() > 10:
                self.detected_positive = False

        if self.detected_positive:
            self.block.set_type(BlockType.Text)

    # Checks geometric and size-based rules to determine if block is text-like
    def is_reasonable_to_be_text(self, block: Block) -> bool:
        if block.get_bounds()[3] > 200:
            return False

        for child in block.get_children():
            if child.get_area() > 50000:
                return False

        children = sorted(block.get_children(), key=lambda b: b.get_bounds()[0] + b.get_bounds()[2])

        for i in range(len(children) - 1):
            first = children[i].get_bounds()
            second = children[i + 1].get_bounds()

            first_min_y = first[1]
            first_max_y = first[1] + first[3]
            second_min_y = second[1]
            second_max_y = second[1] + second[3]

            if first[1] < second[1]:
                if first_max_y - second_min_y < 3:
                    return False
            else:
                if second_max_y - first_min_y < 3:
                    return False

        return True

    # Computes average horizontal spacing between a list of blocks
    def get_average_spacing(self, blocks: list[Block]) -> float:
        if len(blocks) < 2:
            return float('inf')

        blocks_sorted = sorted(blocks, key=lambda b: b.get_bounds()[0])
        total_spacing = 0
        for i in range(len(blocks_sorted) - 1):
            x1 = blocks_sorted[i].get_bounds()[0] + blocks_sorted[i].get_bounds()[2]
            x2 = blocks_sorted[i + 1].get_bounds()[0]
            total_spacing += max(0, x2 - x1)

        return total_spacing / (len(blocks_sorted) - 1)

    # Returns whether the block passed the text detection
    def is_detected_positive(self) -> bool:
        return self.detected_positive

    # Manually override detection result
    def set_detected_positive(self, detected: bool):
        self.detected_positive = detected
