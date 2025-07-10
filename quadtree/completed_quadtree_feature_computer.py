import numpy as np
from quadtree.quadtree_decomposer import QuadTreeDecomposer
from quadtree.quadtree_feature_computer import QuadtreeFeatureComputer
from quadtree.create_binary_mask import create_binary_mask
from quadtree.binary_feature_computer import BinaryFeatureComputer


class FullQuadtreeFeatureComputer:
    def __init__(self, max_depth=5, min_size=20, entropy_threshold=3.0):
        self.decomposer = QuadTreeDecomposer(
            max_depth=max_depth,
            min_size=min_size,
            entropy_threshold=entropy_threshold
        )
        self.features = {}

    def compute_features(self, image: np.ndarray):
        root = self.decomposer.decompose(image)
        self.features['total_nodes'] = QuadtreeFeatureComputer.compute_total_nodes(root)
        self.features['average_entropy'] = QuadtreeFeatureComputer.compute_average_entropy(root)
        self.features['max_depth'] = QuadtreeFeatureComputer.compute_max_depth(root)
        self.features['leaf_area_fraction'] = QuadtreeFeatureComputer.compute_leaf_area_fraction(root)

        mask = create_binary_mask(root, image.shape[:2])
        self.features['horizontal_symmetry'] = BinaryFeatureComputer.compute_horizontal_symmetry(mask)
        self.features['vertical_symmetry'] = BinaryFeatureComputer.compute_vertical_symmetry(mask)
        self.features['horizontal_balance'] = BinaryFeatureComputer.compute_horizontal_balance(mask)
        self.features['vertical_balance'] = BinaryFeatureComputer.compute_vertical_balance(mask)
        self.features['equilibrium'] = BinaryFeatureComputer.compute_equilibrium(mask)

    def is_text(self) -> bool:
        if not self.features:
            raise ValueError("Must call compute_features() first.")

        return (
            self.features['average_entropy'] >= 3.0 and
            self.features['total_nodes'] >= 10 and
            self.features['leaf_area_fraction'] >= 0.5 and
            self.features['horizontal_symmetry'] >= 0.5 and
            self.features['vertical_symmetry'] >= 0.5 and
            self.features['equilibrium'] >= 0.5
        )

    def get_features(self):
        return self.features
