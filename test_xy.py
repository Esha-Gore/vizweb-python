import cv2
import os
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

image_folder = "distinct_images"
output_folder = "output/xy"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    strategy = DefaultXYDecompositionStrategy()
    decomposer = XYDecomposer()
    root = decomposer.decompose(image, strategy)

    # Feature extraction
    avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
    num_leaves = XYFeatureComputer.compute_num_leaves(root)

    print(f"\nImage: {filename}")
    print(f"  Average decomposition depth: {avg_depth}")
    print(f"  Number of leaf blocks: {num_leaves}")

    # Visualization
    visualizer = XYTreeVisualizer()
    outlined = visualizer.draw_block_outlines(image, root)
    cv2.imwrite(os.path.join(output_folder, f"{filename}.png"), outlined)
    print(f"  Saved visualization to {output_folder}/{filename}.png")