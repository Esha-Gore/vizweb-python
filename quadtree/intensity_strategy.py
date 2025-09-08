import numpy as np
import cv2

class IntensityEntropyStrategy:
    def __init__(self, entropy_threshold: float = 3.0):
        # Threshold used to decide if a region has enough intensity variation to be split
        # sidenote: This value may need some fine tuning in Python
        self.entropy_threshold = entropy_threshold

    def should_split(self, region: np.ndarray) -> bool:
        # Compute entropy of the L (lightness) channel and check if it exceeds the threshold
        entropy = self.compute_entropy(region)
        return entropy > self.entropy_threshold

    @staticmethod
    def compute_entropy(image: np.ndarray) -> float:
        # Convert RGB image to L*a*b* color space (L = lightness)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

        # Extract L channel, which represents intensity/lightness
        L_channel = lab[:, :, 0]  # L channel range in OpenCV is 0–255

        # Normalize L to 0–100 range like in Java 
        L_channel = L_channel.astype(np.float32) * (100.0 / 255.0)

        # Compute histogram with 20 bins over range [0, 100]
        hist = cv2.calcHist([L_channel], [0], None, [20], [0, 100])
        hist = hist.ravel()
        hist = hist / hist.sum()  # Normalize histogram to sum to 1

        # Compute Shannon entropy: -Σ p * log₂(p)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add epsilon to avoid log(0)

        return float(entropy)
