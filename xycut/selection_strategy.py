from typing import List
from .separator_model import SeparatorModel

def select_the_largest(separators: List[SeparatorModel]) -> List[SeparatorModel]:
    if not separators:
        return []
    
    # separators = [s for s in separators if s.get_thickness() >= 15]

    best = max(separators, key=lambda s: s.get_width() * s.get_height())
    return [best]

def select_the_largest_and_similar(separators: List[SeparatorModel]) -> List[SeparatorModel]:
    if not separators:
        return []
    
    # separators = [s for s in separators if s.get_thickness() >= 15]


    #Find largest by area, before it was length 
    best = max(separators, key=lambda s: s.get_width() * s.get_height())
    selected = [best]

    for s in separators:
        if s is best:
            continue
        
        # Check thickness similarity (ratio > 0.5)
        min_thick = min(s.get_thickness(), best.get_thickness())
        max_thick = max(s.get_thickness(), best.get_thickness())
        
        if max_thick == 0:
            continue
            
        thickness_ratio = min_thick / max_thick
        is_similar_thickness = thickness_ratio > 0.5
        
        # Check length similarity (within 5 pixels)
        length_diff_ok = abs(s.get_length() - best.get_length()) < 5
        
        if is_similar_thickness and length_diff_ok:
            selected.append(s)

    return selected


