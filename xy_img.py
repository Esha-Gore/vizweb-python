import os
import cv2
from xycut.xy_decomposer import XYDecomposer
from xycut.default_strategy import DefaultXYDecompositionStrategy
from xycut.xy_feature_computer import XYFeatureComputer
from xycut.xycut_visualizer import XYTreeVisualizer

# Input and output directories
image_folder = "/Users/eshagore/Desktop/mini_img"
output_base = "mini_img_xy_output"
os.makedirs(output_base, exist_ok=True)

test_number = 58  # You can change this
debug_base = f"debug/xy/test_{test_number}"
os.makedirs(debug_base, exist_ok=True)

# Get all country folders
country_folders = [f for f in os.listdir(image_folder) 
                   if os.path.isdir(os.path.join(image_folder, f))]

# Process each country folder
for country in sorted(country_folders):
    print(f"Processing country: {country}")
    print("")
    
    country_input = os.path.join(image_folder, country)
    country_output = os.path.join(output_base, country)
    country_debug = os.path.join(debug_base, country)
    
    # Create output directories for this country
    os.makedirs(country_output, exist_ok=True)
    os.makedirs(country_debug, exist_ok=True)
    
    # Get all images in country folder
    images = [f for f in os.listdir(country_input) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(images)} images in {country}")
    
    # Process each image
    for filename in sorted(images):
        image_path = os.path.join(country_input, filename)
        
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  ERROR: Could not read {filename}")
                continue
            
            strategy = DefaultXYDecompositionStrategy()
            strategy.use_space_separators = False
            strategy.line_separator_extractor.debug = True
            
            decomposer = XYDecomposer()
            root = decomposer.decompose(image, strategy)
            
            # Feature extraction
            avg_depth = XYFeatureComputer.compute_average_decomposition_level(root)
            num_leaves = XYFeatureComputer.compute_num_leaves(root)
            
            # Log to country-specific log file
            log_path = os.path.join(country_debug, "log.txt")
            with open(log_path, "a") as f:
                f.write(f"Image {filename}:\n")
                f.write(f"strategy has min_area?: {hasattr(strategy, 'min_area')}\n") 
                f.write(f"strategy.min_area = {getattr(strategy, 'min_area', 'N/A')}\n")
                f.write(f"Average decomposition depth: {avg_depth}\n")
                f.write(f"Number of leaf blocks: {num_leaves}\n\n")
            
            # Visualization - save to country-specific output
            visualizer = XYTreeVisualizer()
            outlined = visualizer.draw_block_outlines(image, root)
            
            output_path = os.path.join(country_output, f"{filename}")
            debug_path = os.path.join(country_debug, f"{filename}")
            
            cv2.imwrite(output_path, outlined)
            cv2.imwrite(debug_path, outlined)
            
            print(f"  ✓ {filename}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            with open(os.path.join(country_debug, "errors.txt"), "a") as f:
                f.write(f"{filename}: {e}\n")
    
    print(f"✓ {country} complete\n")

print("BATCH PROCESSING COMPLETE!")
print(f"Results saved to: {output_base}/")
print(f"Debug logs saved to: {debug_base}/")
