from quadtree.quadtree_node import QuadtreeNode
from typing import Optional


class QuadtreeFeatureComputer:
    @staticmethod
    def compute_total_nodes(node: QuadtreeNode) -> int:
        if node is None:
            return 0
        count = 1
        for child in node.children:
            count += QuadtreeFeatureComputer.compute_total_nodes(child)
        return count

    @staticmethod
    def compute_average_entropy(node: QuadtreeNode) -> float:
        leaves = node.get_all_leaves()
        entropies = [leaf.get_entropy() for leaf in leaves if leaf.get_entropy() is not None]
        return sum(entropies) / len(entropies) if entropies else 0.0

    @staticmethod
    def compute_max_depth(node: QuadtreeNode) -> int:
        if node.is_leaf():
            return 0
        return 1 + max(QuadtreeFeatureComputer.compute_max_depth(c) for c in node.children)

    @staticmethod
    def compute_leaf_area_fraction(node: QuadtreeNode) -> float:
        leaves = node.get_all_leaves()
        total_leaf_area = sum(w * h for (x, y, w, h) in [leaf.get_bounds() for leaf in leaves])
        root_area = node.get_bounds()[2] * node.get_bounds()[3]
        return total_leaf_area / root_area if root_area > 0 else 0.0
