import numpy as np
import cv2
from quadtree.quadtree_node import QuadtreeNode


def draw_quadtree_blocks(image: np.ndarray, root: QuadtreeNode, color=(0, 255, 0), thickness=1) -> np.ndarray:
    """
    Draws the leaf blocks of a quadtree on top of the image.
    Useful for visualizing how the image was partitioned by the quadtree.
    """
    output = image.copy()
    for leaf in root.get_all_leaves():
        x, y, w, h = leaf.get_bounds()
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), color, thickness)
    return output


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.3) -> np.ndarray:
    """
    Overlays a binary mask on top of the image.
    Pixels where the mask is 1 will be tinted with the given color.
    Useful for debugging which regions were marked as content.
    """
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask == 1] = color
    return cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)


def draw_symmetry_lines(image: np.ndarray, color=(255, 0, 0), thickness=1) -> np.ndarray:
    """
    Draws vertical and horizontal symmetry axes (image center lines).
    Useful to visually check symmetry of decomposed regions.
    """
    h, w = image.shape[:2]
    output = image.copy()
    cv2.line(output, (w // 2, 0), (w // 2, h), color, thickness)  # vertical center
    cv2.line(output, (0, h // 2), (w, h // 2), color, thickness)  # horizontal center
    return output


def draw_center_of_mass(image: np.ndarray, mask: np.ndarray, color=(0, 255, 255), radius=4) -> np.ndarray:
    """
    Draws a dot at the center of mass of the active pixels in the mask.
    Useful for equilibrium visualization — how centered the content is.
    """
    output = image.copy()
    if mask.sum() == 0:
        return output
    y_coords, x_coords = np.nonzero(mask)
    cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
    cv2.circle(output, (cx, cy), radius, color, -1)
    return output

def draw_all_blocks(image: np.ndarray, root: QuadtreeNode, color=(0, 255, 0), thickness=3) -> np.ndarray:
    output = image.copy()
    queue = [root]
    while queue:
        node = queue.pop()
        x, y, w, h = node.get_bounds()
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        queue.extend(node.children)
    return output
