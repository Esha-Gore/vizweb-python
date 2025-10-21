from typing import Optional, List, Tuple


class QuadtreeNode:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.bounds: Tuple[int, int, int, int] = (x, y, width, height)
        self.entropy: Optional[float] = None
        self.children: List[QuadtreeNode] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def subdivide(self):
        x, y, w, h = self.bounds

        mid_w = w // 2
        mid_h = h // 2

        # Ensure perfect coverage, even with odd sizes
        children = []

        # Top-left
        children.append(QuadtreeNode(x, y, mid_w, mid_h))

        # Top-right
        children.append(QuadtreeNode(x + mid_w, y, w - mid_w, mid_h))

        # Bottom-left
        children.append(QuadtreeNode(x, y + mid_h, mid_w, h - mid_h))

        # Bottom-right
        children.append(QuadtreeNode(x + mid_w, y + mid_h, w - mid_w, h - mid_h))

        self.children = children


    def get_all_leaves(self) -> List['QuadtreeNode']:
        if self.is_leaf():
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves.extend(child.get_all_leaves())
            return leaves
        
    def num_leaves(self) -> int:
        leaves = self.get_all_leaves()
        return len(leaves)

    def set_entropy(self, value: float):
        self.entropy = value

    def get_entropy(self) -> Optional[float]:
        return self.entropy

    def get_bounds(self) -> Tuple[int, int, int, int]:
        return self.bounds
