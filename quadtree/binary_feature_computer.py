import numpy as np
import cv2

class BinaryFeatureComputer:

    # Compares the full mask to its vertical flip (top/bottom symmetry).
    # Returns 1.0 for perfect symmetry, 0.0 for complete asymmetry.
    @staticmethod
    def compute_horizontal_symmetry(mask: np.ndarray) -> float:
        flipped = np.flipud(mask)
        diff = cv2.absdiff(mask.astype(np.uint8), flipped.astype(np.uint8))
        non_symmetric_pixels = np.count_nonzero(diff)
        total_pixels = mask.size
        return 1.0 - (non_symmetric_pixels / total_pixels) if total_pixels else 0.0

    # Compares the full mask to its horizontal flip (left/right symmetry).
    # Returns 1.0 for perfect symmetry, 0.0 for complete asymmetry.
    @staticmethod
    def compute_vertical_symmetry(mask: np.ndarray) -> float:
        flipped = np.fliplr(mask)
        diff = cv2.absdiff(mask.astype(np.uint8), flipped.astype(np.uint8))
        non_symmetric_pixels = np.count_nonzero(diff)
        total_pixels = mask.size
        return 1.0 - (non_symmetric_pixels / total_pixels) if total_pixels else 0.0

    # Computes vertical center of mass (how high or low the content is).
    # Returns a value between 0.0 (top) and 1.0 (bottom).
    @staticmethod
    def compute_horizontal_balance(mask: np.ndarray) -> float:
        h, w = mask.shape
        weights = np.arange(h).reshape(-1, 1)
        mass = mask.sum()
        if mass == 0:
            return 0.0
        balance = np.sum(weights * mask) / mass
        return balance / h

    # Computes horizontal center of mass (how left or right the content is).
    # Returns a value between 0.0 (left) and 1.0 (right).
    @staticmethod
    def compute_vertical_balance(mask: np.ndarray) -> float:
        h, w = mask.shape
        weights = np.arange(w).reshape(1, -1)
        mass = mask.sum()
        if mass == 0:
            return 0.0
        balance = np.sum(weights * mask) / mass
        return balance / w

    # Measures how close the center of mass is to the image center.
    # Returns 1.0 if perfectly centered, 0.0 if it's at the corner.
    @staticmethod
    def compute_equilibrium(mask: np.ndarray) -> float:
        h, w = mask.shape
        if mask.sum() == 0:
            return 0.0
        y_coords, x_coords = np.nonzero(mask)
        cy = np.mean(y_coords)
        cx = np.mean(x_coords)
        distance = np.sqrt((cy - h/2) ** 2 + (cx - w/2) ** 2)
        max_distance = np.sqrt((h/2) ** 2 + (w/2) ** 2)
        return 1.0 - (distance / max_distance)

