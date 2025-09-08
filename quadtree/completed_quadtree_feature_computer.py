import numpy as np
from quadtree.quadtree_decomposer import QuadTreeDecomposer
from quadtree.quadtree_feature_computer import QuadtreeFeatureComputer
from quadtree.create_binary_mask import create_binary_mask
from quadtree.binary_feature_computer import BinaryFeatureComputer


class FullQuadtreeFeatureComputer:
    #Change the entropy threshold here:
    def __init__(self, max_depth=5, min_size=20, entropy_threshold=2.2):
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

    def get_features(self):
        return self.features
