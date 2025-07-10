from typing import List
from collections import deque
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_text_detector import XYTextDetector
from xycut.block import Block

# The purpose of this file is to compute the spacetree features of an image,
# including xy block structure, non-text block area, percentage of area 
# covered by leaves (as a check), average depth of the tree, and maximum depth. 
class XYFeatureComputer:

    # returns the root of a block tree containing detected text regions,
    # after filtering out small/noisy blocks and removing children of text blocks.
    @staticmethod
    def get_xy_block_structure(image) -> Block:
        decomposer = XYDecomposer()
        root = decomposer.decompose(image, DefaultXYDecompositionStrategy())
        root.filter_out_small_blocks()

        detector = XYTextDetector(root, image)
        root_with_text = detector.detect()
        root_with_text.remove_children_of_text_blocks()

        return root_with_text

    # returns the number of pixels of non-text blocks. 
    @staticmethod
    def compute_non_text_block_area(node: Block) -> int:
        if node.is_leaf():
            return node.get_area() if not node.is_text() else 0
        return sum(XYFeatureComputer.compute_non_text_block_area(c) for c in node.get_children())

    # returns the percentage of the root block's area that is covered by its leaf blocks.
    # sidenote: in a properly decomposed tree this should return 100%.
    @staticmethod
    def compute_percentage_of_leaf_area(root: Block) -> float:
        leaves = XYFeatureComputer.get_all_leaves(root)
        total_leaf_area = sum(leaf.get_area() for leaf in leaves)
        total_area = root.get_area()
        return 100.0 * total_leaf_area / total_area if total_area > 0 else 0.0

    # returns the percentage of area covered by all leaf blocks relative to the total area.
    # sidenote: if the root has only one child, the child's area is used as the base.
    @staticmethod
    def compute_percentage_of_leaf_area2(root: Block) -> float:
        leaves = XYFeatureComputer.get_all_leaves(root)
        total_leaf_area = sum(leaf.get_area() for leaf in leaves)
        total_area = root.get_area()
        if XYFeatureComputer.count_number_of_nodes_in_level(root, 1) == 1 and root.get_first_child():
            total_area = root.get_first_child().get_area()
        return 100.0 * total_leaf_area / total_area if total_area > 0 else 0.0

    # returns the average depth of all leaf nodes in the tree
    @staticmethod
    def compute_average_decomposition_level(root: Block) -> float:
        leaves = XYFeatureComputer.get_all_leaves(root)
        depths = [XYFeatureComputer.compute_node_depth(leaf) for leaf in leaves]
        return sum(depths) / len(depths) if depths else 0.0

    # returns the depth of a given node. 
    @staticmethod
    def compute_node_depth(node: Block) -> int:
        depth = 0
        current = node.get_parent()
        while current:
            depth += 1
            current = current.get_parent()
        return depth

    # returns a list of all leaf nodes in the block tree using a breadth-first search.
    @staticmethod
    def get_all_leaves(root: Block) -> List[Block]:
        if root is None:
            return []
        leaves = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.is_leaf():
                leaves.append(node)
            queue.extend(node.get_children())
        return leaves

    # returns the maximum depth of the subtree with the given root. 
    @staticmethod
    def compute_maximum_decomposition_level(root: Block) -> int:
        return XYFeatureComputer.compute_node_height(root)

    # returns the height of the given node based on its subtree. 
    @staticmethod
    def compute_node_height(node: Block) -> int:
        if node.is_leaf():
            return 0
        return 1 + max(XYFeatureComputer.compute_node_height(c) for c in node.get_children())

    # returns the total text area of the given root's subtree. 
    @staticmethod
    def compute_text_area(root: Block) -> int:
        return XYFeatureComputer.compute_text_area_recursive(root)

    # helper method which computes the total text area of a root recursively. 
    @staticmethod
    def compute_text_area_recursive(node: Block) -> int:
        if node.is_leaf():
            return node.get_area() if node.is_text() else 0
        return sum(XYFeatureComputer.compute_text_area_recursive(c) for c in node.get_children())

    # returns the number of nodes in the same depth-level based on a given tree's root and given level
    @staticmethod
    def count_number_of_nodes_in_level(root: Block, level: int) -> int:
        return XYFeatureComputer.recursive_count_nodes_in_level(root, level, 0)

    # helper method to compute number of nodes in a level recursively. 
    @staticmethod
    def recursive_count_nodes_in_level(node: Block, level: int, current: int) -> int:
        if current == level:
            return 1
        return sum(XYFeatureComputer.recursive_count_nodes_in_level(c, level, current + 1) for c in node.get_children())

    # returns the number of non-leaf nodes whose children have text groups. 
    @staticmethod
    def count_number_of_text_groups(root: Block) -> int:
        return XYFeatureComputer.recursive_count_text_groups(root)
    
    # helper recursive method. 
    @staticmethod
    def recursive_count_text_groups(node: Block) -> int:
        if node.is_leaf():
            return 0
        count = 1 if XYFeatureComputer.block_contains_text_children(node) else 0
        count += sum(XYFeatureComputer.recursive_count_text_groups(c) for c in node.get_children())
        return count
    # returns whether the root has any text block children.
    @staticmethod
    def block_contains_text_children(node: Block) -> bool:
        return any(child.is_text_block() for child in node.get_children())

    # returns the number of non-leaf nodes whose children include image-containing blocks.
    @staticmethod
    def count_number_of_image_areas(root: Block) -> int:
        return XYFeatureComputer.recursive_count_image_areas(root)

    # helper recursive method
    @staticmethod
    def recursive_count_image_areas(node: Block) -> int:
        if node.is_leaf():
            return 0
        count = 1 if XYFeatureComputer.block_contains_image_children(node) else 0
        count += sum(XYFeatureComputer.recursive_count_image_areas(c) for c in node.get_children())
        return count

    # returns whether the given block has any children which contain images. 
    @staticmethod
    def block_contains_image_children(node: Block) -> bool:
        return any(c.is_leaf() and not c.is_text_block() and c.get_area() >= 800 for c in node.get_children())

    # returns the area of all non-text leaves. 
    @staticmethod
    def compute_non_text_leaves_area(root: Block) -> int:
        return XYFeatureComputer.compute_non_text_leaves_recursive(root)

    # helper method
    @staticmethod
    def compute_non_text_leaves_recursive(node: Block) -> int:
        if node.is_leaf():
            return node.get_area() if not node.is_text() else 0
        return sum(XYFeatureComputer.compute_non_text_leaves_recursive(c) for c in node.get_children())
