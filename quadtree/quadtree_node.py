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
        half_w = w // 2
        half_h = h // 2

        self.children = [
            QuadtreeNode(x, y, half_w, half_h),
            QuadtreeNode(x + half_w, y, w - half_w, half_h),
            QuadtreeNode(x, y + half_h, half_w, h - half_h),
            QuadtreeNode(x + half_w, y + half_h, w - half_w, h - half_h)
        ]

    def get_all_leaves(self) -> List['QuadtreeNode']:
        if self.is_leaf():
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves.extend(child.get_all_leaves())
            return leaves

    def set_entropy(self, value: float):
        self.entropy = value

    def get_entropy(self) -> Optional[float]:
        return self.entropy

    def get_bounds(self) -> Tuple[int, int, int, int]:
        return self.bounds
