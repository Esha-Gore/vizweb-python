import cv2
import numpy as np
from typing import Tuple

# Represents the visual line that splits the regions in the XYCUT Metric
# Can be used to draw them onto the image. 
class SeparatorModel:
    HORIZONTAL_SEPARATOR = 1
    VERTICAL_SEPARATOR = 2

    def __init__(self, position=None, direction=None, roi=None, bounds=None):
        self.position = position
        self.direction = direction
        self.color = (0, 255, 0)
        self.roi = roi

        if bounds:
            self.bounds = bounds
        elif roi:
            self.bounds = (
                roi[0],
                roi[1],
                1 if direction == self.VERTICAL_SEPARATOR else roi[2],
                1 if direction == self.HORIZONTAL_SEPARATOR else roi[3]
            )

    def get_height(self):
        return self.bounds[3]

    def get_width(self):
        return self.bounds[2]

    def get_x(self):
        return self.bounds[0]

    def get_y(self):
        return self.bounds[1]

    def get_location(self):
        return (self.get_x(), self.get_y())

    def get_bounds(self):
        return self.bounds

    def is_horizontal(self):
        return self.direction == self.HORIZONTAL_SEPARATOR

    def is_vertical(self):
        return self.direction == self.VERTICAL_SEPARATOR

    def get_length(self):
        return self.get_width() if self.is_horizontal() else self.get_height()

    def get_thickness(self):
        return self.get_height() if self.is_horizontal() else self.get_width()

    def set_color(self, color: Tuple[int, int, int]):
        self.color = color

    def get_color(self):
        return self.color

    def set_position(self, position: int):
        self.position = position

    def paint(self, image: np.ndarray) -> np.ndarray:
        x, y, w, h = self.bounds
        roi = image[y:y+h, x:x+w]
        roi[:] = self.color  # Fill ROI with color

        # Draw rectangle border on the original image
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)  # Blue border

        return image
