import numpy as np


class BinaryFeatureComputer:

    @staticmethod
    def compute_horizontal_symmetry(mask: np.ndarray) -> float:
        h = mask.shape[0]
        top = mask[:h // 2, :]
        bottom = np.flipud(mask[h - h // 2:, :])
        diff = np.abs(top - bottom)
        return 1.0 - (np.sum(diff) / np.sum(mask)) if np.sum(mask) else 0.0

    @staticmethod
    def compute_vertical_symmetry(mask: np.ndarray) -> float:
        w = mask.shape[1]
        left = mask[:, :w // 2]
        right = np.fliplr(mask[:, w - w // 2:])
        diff = np.abs(left - right)
        return 1.0 - (np.sum(diff) / np.sum(mask)) if np.sum(mask) else 0.0

    @staticmethod
    def compute_horizontal_balance(mask: np.ndarray) -> float:
        h, w = mask.shape
        weights = np.arange(h).reshape(-1, 1)
        mass = mask.sum()
        if mass == 0:
            return 0.0
        balance = np.sum(weights * mask) / mass
        return balance / h

    @staticmethod
    def compute_vertical_balance(mask: np.ndarray) -> float:
        h, w = mask.shape
        weights = np.arange(w).reshape(1, -1)
        mass = mask.sum()
        if mass == 0:
            return 0.0
        balance = np.sum(weights * mask) / mass
        return balance / w

    @staticmethod
    def compute_equilibrium(mask: np.ndarray) -> float:
        h, w = mask.shape
        cy, cx = np.array(np.nonzero(mask)).mean(axis=1, initial=0) if mask.sum() else (0, 0)
        return 1.0 - np.sqrt(((cy - h/2) ** 2 + (cx - w/2) ** 2)) / np.sqrt((h/2) ** 2 + (w/2) ** 2)
