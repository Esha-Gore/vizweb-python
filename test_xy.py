import cv2
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

image = cv2.imread("images/5935268355cac.jpg")
strategy = DefaultXYDecompositionStrategy()

decomposer = XYDecomposer()
root = decomposer.decompose(image, strategy)
print("strategy has min_area?", hasattr(strategy, "min_area"))
print("strategy.min_area =", getattr(strategy, "min_area", "N/A"))


# Feature extraction
avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
num_leaves = XYFeatureComputer.compute_num_leaves(root)

# Visualization
visualizer = XYTreeVisualizer()
outlined = visualizer.draw_block_outlines(image, root)
cv2.imwrite("output_visuals/vis_sample.png", outlined)

print(f"Average decomposition depth: {avg_depth}")
print(f"Number of leaf blocks: {num_leaves}")