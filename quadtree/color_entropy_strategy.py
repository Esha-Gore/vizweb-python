import numpy as np
import cv2

class ColorEntropyStrategy:

    # Threshold used to decide if a region has enough color variation to be split
    # **** threshold mirrors the Java default, but may need tuning in Python
    def __init__(self, entropy_threshold: float = 300.0):
        self.entropy_threshold = entropy_threshold

    # Computes color entropy and checks if it's above the threshold
    def should_split(self, region: np.ndarray) -> bool:
        entropy = self.compute_entropy(region)
        return entropy > self.entropy_threshold

    @staticmethod
    def compute_entropy(image: np.ndarray) -> float:
        # Convert the RGB image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Split into H (Hue) and S (Saturation) channels
        h = hsv[:, :, 0]  # Hue: 0–180
        s = hsv[:, :, 1]  # Saturation: 0–255

        # Compute 2D histogram for (H, S) with same bin settings as Java
        hist = cv2.calcHist([h, s], [0, 1], None, [30, 32], [0, 180, 0, 256])

        # Normalize to convert counts to probabilities
        hist = hist / hist.sum()

        # Compute joint Shannon entropy: -Σ p * log2(p)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Avoid log(0) ??? idk this screws things up something

        return float(entropy)

