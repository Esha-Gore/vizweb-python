import cv2
import os
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

#from pdb import set_trace as st
image_folder = "distinct_images"
os.makedirs("debug", exist_ok=True)
os.makedirs("debug_output", exist_ok=True)


# todo: hyperparameter search first 
# then conside the defautl strategy pixels (like by 100 or so)
# choose something which makes sense (things are wider then they are long, etc)
# evalute them. 

# commit code, and create READ me (env instructions and testing instructions)

test_number = 9
os.makedirs(f"debug/xy", exist_ok=True)
os.makedirs(f"debug/xy/test_{test_number}", exist_ok=True)

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)

    strategy = DefaultXYDecompositionStrategy()

    decomposer = XYDecomposer()
    root = decomposer.decompose(image, strategy)
    # print("strategy has min_area?", hasattr(strategy, "min_area"))
    # print("strategy.min_area =", getattr(strategy, "min_area", "N/A"))


    # Feature extraction
    avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
    num_leaves = XYFeatureComputer.compute_num_leaves(root)

    # print(f"Average decomposition depth: {avg_depth}")
    # print(f"Number of leaf blocks: {num_leaves}")

    with open(f"debug/xy/test_{test_number}/log.txt", "a") as f:
        f.write(f"Image {filename}:\n")
        f.write(f"strategy has min_area?,{hasattr(strategy, "min_area")}\n")
        f.write(f"strategy.min_area = {getattr(strategy, "min_area", "N/A")}\n")
        f.write(f"Average decomposition depth: {avg_depth}\n")
        f.write(f"Number of leaf blocks: {num_leaves}\n\n")

    # Visualization
    visualizer = XYTreeVisualizer()
    outlined = visualizer.draw_block_outlines(image, root)
    cv2.imwrite(f"debug/xy/test_{test_number}/{filename}.png", outlined)
    print(f"done:{filename}")
    # cv2.imshow("Spacetree Visualization", outlined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
