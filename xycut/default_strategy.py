from typing import List, Callable
from xycut.separator_model import SeparatorModel
from xycut.separator_extractor import SeparatorExtractor
from xycut.space_analyzer import find_content_bounds
from xycut.line_separator_extractor import LineSeparatorExtractor
from xycut.space_separator_extractor import SpaceSeparatorExtractor
from xycut.decomposition_strategy import DecompositionStrategy
from xycut.selection_strategy import select_the_largest, select_the_largest_and_similar
import cv2
import numpy as np

class DefaultXYDecompositionStrategy(DecompositionStrategy):
    def __init__(
        self,
        # change from select the largest & most similar to just teh largest
        separator_selection_strategy: Callable[[List[SeparatorModel]], List[SeparatorModel]] = select_the_largest_and_similar,
        min_separator_size: int = 10,
        min_std_dev: int = 10,
        # fine tune 
        min_area: int = 100,
        min_width: int = 40,
        min_height: int = 20,
        max_level: int = 15,
    remove_border: bool = True,
        split_horizontal: bool = True,
        split_vertical: bool = True,
        use_line_separators: bool = True,
        use_space_separators: bool = True,
        line_separator_extractor: SeparatorExtractor = None,
        space_separator_extractor: SeparatorExtractor = None,
        debug: bool = False
    ):
        self.separator_selection_strategy = separator_selection_strategy
        self.min_separator_size = min_separator_size
        self.min_std_dev = min_std_dev
        self.min_area = min_area
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

    def choose_separators(self, separators: List[SeparatorModel]) -> List[SeparatorModel]:
        return self.separator_selection_strategy(separators)

    def find_content_bounds(self, image, bounds):
        return find_content_bounds(image, bounds)

    def is_splitting_further(self, separator: SeparatorModel, level: int) -> bool:
        return separator is not None and level < self.max_level

    def get_separator_selection_strategy(self) -> Callable[[List[SeparatorModel]], List[SeparatorModel]]:
        return self.separator_selection_strategy

    def visualize_separators(self, image, bounds, separators: List[SeparatorModel], level: int):
        if not self.debug:
            return

        x, y, w, h = bounds
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        for sep in separators:
            sx, sy, sw, sh = sep.get_bounds()
            cv2.rectangle(vis_image, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_image, f"Level {level}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(f"Separators at level {level}", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



