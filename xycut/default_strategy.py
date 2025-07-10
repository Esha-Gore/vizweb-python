from typing import List, Callable
from xycut.separator_model import SeparatorModel
from xycut.separator_extractor import SeparatorExtractor
from xycut.space_analyzer import find_content_bounds
from xycut.line_separator_extractor import LineSeparatorExtractor
from xycut.space_separator_extractor import SpaceSeparatorExtractor
import cv2
import numpy as np

class DefaultXYDecompositionStrategy:
    class SeparatorSelectionStrategies:
        # Selects the separator with the largest area
        @staticmethod
        def select_the_largest(separators: List[SeparatorModel]) -> List[SeparatorModel]:
            if not separators:
                return []
            best = max(separators, key=lambda s: s.get_width() * s.get_height())
            return [best]

        # Selects the largest separator and any similar in thickness and length
        @staticmethod
        def select_the_largest_and_similar(separators: List[SeparatorModel]) -> List[SeparatorModel]:
            if not separators:
                return []
            best = max(separators, key=lambda s: s.get_width() * s.get_height())
            selected = [best]
            for s in separators:
                if s is best:
                    continue
                thickness_ratio = min(s.get_thickness(), best.get_thickness()) / max(s.get_thickness(), best.get_thickness())
                length_diff_ok = abs(s.get_length() - best.get_length()) < 5
                if thickness_ratio > 0.5 and length_diff_ok:
                    selected.append(s)
            return selected

    # Initializes the decomposition strategy with thresholds, flags, and tools
def __init__(
    self,
    separator_selection_strategy: Callable[[List[SeparatorModel]], List[SeparatorModel]] = None,
    min_separator_size: int = 10,
    min_std_dev: int = 10,
    min_area: int = 50,
    min_width: int = 5,
    min_height: int = 5,
    max_level: int = 10,
    remove_border: bool = True,
    split_horizontal: bool = True,
    split_vertical: bool = True,
    use_line_separators: bool = True,
    use_space_separators: bool = True,
    line_separator_extractor: SeparatorExtractor = None,
    space_separator_extractor: SeparatorExtractor = None,
    debug: bool = False
):
    self.separator_selection_strategy = separator_selection_strategy or self.SeparatorSelectionStrategies.select_the_largest_and_similar
    self.min_separator_size = min_separator_size
    self.min_std_dev = min_std_dev
    self.min_area = min_area
    print("min_area:", min_area)
    self.min_width = min_width
    self.min_height = min_height
    self.max_level = max_level
    self.remove_border = remove_border
    self.split_horizontal = split_horizontal
    self.split_vertical = split_vertical

    self.use_line_separators = use_line_separators
    self.use_space_separators = use_space_separators

    self.line_separator_extractor = line_separator_extractor or SeparatorExtractor()
    self.space_separator_extractor = space_separator_extractor or SeparatorExtractor()
    self.debug = debug

    # Applies the current selection strategy to filter separators
    def choose_separators(self, separators: List[SeparatorModel]) -> List[SeparatorModel]:
        return self.separator_selection_strategy(separators)

    # Trims whitespace margins from the given region of the image
    def find_content_bounds(self, image, bounds):
        return find_content_bounds(image, bounds)

    # Determines whether to continue splitting at the given level
    def is_splitting_further(self, separator: SeparatorModel, level: int) -> bool:
        return separator is not None and level < self.max_level

    # Returns the active separator selection strategy function
    def get_separator_selection_strategy(self) -> Callable[[List[SeparatorModel]], List[SeparatorModel]]:
        return self.separator_selection_strategy

    # Visualizes the separators and ROI at the current decomposition level
    def visualize_separators(self, image, bounds, separators: List[SeparatorModel], level: int):
        if not self.debug:
            return

        x, y, w, h = bounds
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        for sep in separators:
            sx, sy, sw, sh = sep.get_bounds()
            color = (0, 0, 255)  # red
            cv2.rectangle(vis_image, (sx, sy), (sx + sw, sy + sh), color, 2)

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green for ROI
        cv2.putText(vis_image, f"Level {level}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(f"Separators at level {level}", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



