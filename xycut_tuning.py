import cv2
import os
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

#from pdb import set_trace as st
image_folder = "distinct_images"
os.makedirs("xy_cut_hyperparameter_tuning/T2", exist_ok=True)
os.makedirs("xy_cut_hyperparameter_tuning/T2/c_threshold", exist_ok=True)

values= [15, 18, 20, 22, 25, 30, 35]

for i in values:

    os.makedirs(f"xy_cut_hyperparameter_tuning/T2/c_threshold/{i}", exist_ok=True)

    for filename in os.listdir(image_folder):
        #filename = "pelican.jpg"
        #filename = "youtube.png"
        image_path = os.path.join(image_folder, filename)

        image = cv2.imread(image_path)

        strategy = DefaultXYDecompositionStrategy()

        decomposer = XYDecomposer(c_threshold= i)
        root = decomposer.decompose(image, strategy)
        # Feature extraction
        avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
        num_leaves = XYFeatureComputer.compute_num_leaves(root)

        with open(f"xy_cut_hyperparameter_tuning/T2/c_threshold/{i}/log.txt", "a") as f:
            f.write(f"Image {filename}:\n")
            f.write(f"strategy has min_area?,{hasattr(strategy, 'min_area') }\n") 
            f.write(f"strategy.min_area = {getattr(strategy, 'min_area', 'N/A')}\n")
            f.write(f"Average decomposition depth: {avg_depth}\n")
            f.write(f"Number of leaf blocks: {num_leaves}\n\n")

        # Visualization
        visualizer = XYTreeVisualizer()
        outlined = visualizer.draw_block_outlines(image, root)
        cv2.imwrite(f"xy_cut_hyperparameter_tuning/T2/c_threshold/{i}/{filename}.png", outlined)
        print(f"done:{filename}")
