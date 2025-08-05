from typing import List, Tuple, Optional
from xycut.block_type import BlockType

class Block:
    # MINIMUM_AREA = 10
    # MINIMUM_WIDTH = 2
    # MINIMUM_HEIGHT = 2
    MINIMUM_AREA = 500
    MINIMUM_WIDTH = 500
    MINIMUM_HEIGHT = 500

    def __init__(self):
        self.bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.parent: Optional["Block"] = None
        self.children: List["Block"] = []
        self.type: str = BlockType.Unknown
        self.properties: dict = {}

    def set_bounds(self, bounds: Tuple[int, int, int, int]):
        self.bounds = bounds

    def add_child(self, child: "Block"):
        child.parent = self
        self.children.append(child)

    def get_children(self) -> List["Block"]:
        return self.children

    def get_first_child(self) -> Optional["Block"]:
        return self.children[0] if self.children else None
    
    def get_bounds(self):
        return self.bounds

    def get_second_child(self) -> Optional["Block"]:
        return self.children[1] if len(self.children) >= 2 else None

    def get_child(self, index: int) -> Optional["Block"]:
        return self.children[index] if index < len(self.children) else None

    def get_parent(self) -> Optional["Block"]:
        return self.parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_text_block(self) -> bool:
        return self.type == BlockType.Text

    def is_text(self) -> bool:
        if self.is_text_block() and self.children:
            self.children.clear()
        return self.is_text_block()

    def set_type(self, block_type: str):
        self.type = block_type

    def get_type(self) -> str:
        return self.type

    def get_area(self) -> int:
        _, _, w, h = self.bounds
        return w * h

    def get_leaves(self) -> List["Block"]:
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def add_property(self, key, value):
        self.properties[key] = value

    def get_property(self, key):
        return self.properties.get(key)

    def filter_out_small_blocks(self):
        self.children = [
            c for c in self.children
            if c.get_area() >= self.MINIMUM_AREA and
               c.bounds[2] >= self.MINIMUM_WIDTH and
               c.bounds[3] >= self.MINIMUM_HEIGHT
        ]
        for c in self.children:
            c.filter_out_small_blocks()

    def remove_children_of_text_blocks(self):
        if self.is_leaf():
            return
        if self.is_text_block():
            self.children.clear()
        else:
            for c in self.children:
                c.remove_children_of_text_blocks()

    def get_x(self) -> int:
        return self.bounds[0]

    def get_y(self) -> int:
        return self.bounds[1]

    def get_width(self) -> int:
        return self.bounds[2]

    def get_height(self) -> int:
        return self.bounds[3]


