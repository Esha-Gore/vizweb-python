import cv2
import os
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

#from pdb import set_trace as st
image_folder = "hyperparameter"
os.makedirs("xy_cut_hyperparameter_tuning", exist_ok=True)
os.makedirs("xy_cut_hyperparameter_tuning/spatial", exist_ok=True)

for filename in os.listdir(image_folder):
    #filename = "pelican.jpg"
    #filename = "youtube.png"
    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)

    strategy = DefaultXYDecompositionStrategy()

    decomposer = XYDecomposer()
    root = decomposer.decompose(image, strategy)
    # Feature extraction
    avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
    num_leaves = XYFeatureComputer.compute_num_leaves(root)

    # Visualization
    visualizer = XYTreeVisualizer()
    outlined = visualizer.draw_block_outlines(image, root)
    cv2.imwrite(f"xy_cut_hyperparameter_tuning/spatial/{filename}_4.png", outlined)
    print(f"done:{filename}")
