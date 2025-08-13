from typing import List
from .separator_model import SeparatorModel

def select_the_largest(separators: List[SeparatorModel]) -> List[SeparatorModel]:
    if not separators:
        return []
    best = max(separators, key=lambda s: s.get_width() * s.get_height())
    return [best]

def select_the_largest_and_similar(separators: List[SeparatorModel]) -> List[SeparatorModel]:
    if not separators:
        return []

    def length(s):
        return s.get_length()

    best = max(separators, key=length)
    selected = [best]

    for s in separators:
        if s is best:
            continue
        thickness_ratio = min(s.get_thickness(), best.get_thickness()) / max(s.get_thickness(), best.get_thickness())
        length_diff_ok = abs(s.get_length() - best.get_length()) < 5
        if thickness_ratio > 0.5 and length_diff_ok:
            selected.append(s)

    return selected

