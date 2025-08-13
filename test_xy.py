import cv2
import os
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

#from pdb import set_trace as st
image_folder = "images"
os.makedirs("debug", exist_ok=True)
os.makedirs("debug_output", exist_ok=True)


for filename in os.listdir(image_folder):
    #filename = "pelican.jpg"
    filename = "593525f07e402.jpg"
    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)

    strategy = DefaultXYDecompositionStrategy()

    decomposer = XYDecomposer()
    root = decomposer.decompose(image, strategy)
    print("strategy has min_area?", hasattr(strategy, "min_area"))
    print("strategy.min_area =", getattr(strategy, "min_area", "N/A"))


    # Feature extraction
    avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
    num_leaves = XYFeatureComputer.compute_num_leaves(root)

    print(f"Average decomposition depth: {avg_depth}")
    print(f"Number of leaf blocks: {num_leaves}")

    # Visualization
    visualizer = XYTreeVisualizer()
    outlined = visualizer.draw_block_outlines(image, root)
    cv2.imshow("Spacetree Visualization", outlined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
