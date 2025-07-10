import cv2
from xycut.xy_decomposer import XYDecomposer
from xycut.default_xy_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xy_tree_visualizer import XYTreeVisualizer

image = cv2.imread("images/sample.png")
strategy = DefaultXYDecompositionStrategy()

decomposer = XYDecomposer()
root = decomposer.decompose(image, strategy)

# Feature extraction
avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
num_leaves = XYFeatureComputer.compute_num_leaves(root)

# Visualization
visualizer = XYTreeVisualizer()
outlined = visualizer.draw_block_outlines(image, root)
cv2.imwrite("output_visuals/vis_sample.png", outlined)

print(f"Average decomposition depth: {avg_depth}")
print(f"Number of leaf blocks: {num_leaves}")
