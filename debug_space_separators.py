import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List
import sys

from xycut.space_separator_extractor import SpaceSeparatorExtractor
from xycut.block import Block

class DebugSpaceSeparatorExtractor:
    """
    Enhanced version of SpaceSeparatorExtractor with detailed debugging output
    """
    
    def __init__(self, base_std_threshold: int = 10):
        self.base_std_threshold = base_std_threshold

    def debug_extract(self, block_bounds, image, h_min_thickness=15, v_min_thickness=15, 
                     r_threshold=35, c_threshold=30, save_prefix="debug"):
        """
        Extract separators with detailed debugging visualizations
        
        Args:
            block_bounds: (x, y, width, height) tuple
            image: BGR image
            save_prefix: prefix for saved debug images
        """
        bx, by, bw, bh = block_bounds
        
        print(f"\n{'='*60}")
        print(f"DEBUGGING BLOCK: ({bx}, {by}, {bw}, {bh})")
        print(f"{'='*60}\n")
        
        # Extract ROI
        roi_bgr = image[by:by+bh, bx:bx+bw]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Show original
        self._save_image(roi_bgr, f"{save_prefix}_01_original.png")
        self._save_image(gray, f"{save_prefix}_02_grayscale.png")
        
        # Light blur
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        self._save_image(gray_blurred, f"{save_prefix}_03_blurred.png")
        
        # Calculate standard deviation
        row_std = np.std(gray_blurred, axis=1)
        col_std = np.std(gray_blurred, axis=0)
        
        print(f"Row std - min: {row_std.min():.2f}, max: {row_std.max():.2f}, mean: {row_std.mean():.2f}")
        print(f"Col std - min: {col_std.min():.2f}, max: {col_std.max():.2f}, mean: {col_std.mean():.2f}")
        
        # Plot std distributions
        self._plot_std_distribution(row_std, col_std, save_prefix)
        
        # Calculate thresholds
        col_threshold = min(max(self.base_std_threshold, np.percentile(col_std, 20)), c_threshold)
        row_threshold = min(max(self.base_std_threshold, np.percentile(row_std, 20)), r_threshold)
        
        print(f"\nThresholds:")
        print(f"  Row threshold: {row_threshold:.2f}")
        print(f"  Col threshold: {col_threshold:.2f}")

        row_threshold = max(row_threshold, 25)
        col_threshold = max(col_threshold, 25)
        
        # Create masks
        horizontal_mask = row_std < row_threshold
        vertical_mask = col_std < col_threshold
        
        print(f"\nMask statistics:")
        print(f"  Horizontal mask: {np.sum(horizontal_mask)}/{len(horizontal_mask)} rows flagged as 'empty'")
        print(f"  Vertical mask: {np.sum(vertical_mask)}/{len(vertical_mask)} cols flagged as 'empty'")
        
        # Visualize masks
        self._visualize_masks(gray_blurred, horizontal_mask, vertical_mask, 
                             row_threshold, col_threshold, save_prefix)
        
        # Find segments
        horizontal_segments = self._segment_boolean_mask(horizontal_mask)
        vertical_segments = self._segment_boolean_mask(vertical_mask)
        
        print(f"\nSegments found:")
        print(f"  Horizontal segments: {len(horizontal_segments)}")
        print(f"  Vertical segments: {len(vertical_segments)}")
        
        # Analyze each segment
        print(f"\n{'='*60}")
        print("HORIZONTAL SEGMENTS (potential horizontal separators):")
        print(f"{'='*60}")
        self._analyze_segments(horizontal_segments, row_std, gray_blurred, 
                              is_horizontal=True, min_thickness=h_min_thickness, 
                              block_height=bh)
        
        print(f"\n{'='*60}")
        print("VERTICAL SEGMENTS (potential vertical separators):")
        print(f"{'='*60}")
        self._analyze_segments(vertical_segments, col_std, gray_blurred, 
                              is_horizontal=False, min_thickness=v_min_thickness,
                              block_width=bw)
        
        # Visualize all separators on original image
        self._visualize_separators(roi_bgr, horizontal_segments, vertical_segments,
                                   h_min_thickness, v_min_thickness, bw, bh, save_prefix)
        
        print(f"\n{'='*60}")
        print(f"Debug images saved with prefix: {save_prefix}")
        print(f"{'='*60}\n")

    def _analyze_segments(self, segments, std_values, gray_image, is_horizontal, 
                         min_thickness, block_height=None, block_width=None):
        """Analyze each segment in detail"""
        
        dimension = block_height if is_horizontal else block_width
        min_thick = max(min_thickness, int(0.02 * dimension)) if dimension else min_thickness
        
        for i, seg in enumerate(segments):
            start = seg['start']
            length = seg['length']
            
            print(f"\n  Segment #{i+1}:")
            print(f"    Position: {start}, Length: {length}")
            print(f"    Min thickness required: {min_thick}")
            
            # Check if it passes thickness filter
            if length < min_thick:
                print(f"    REJECTED: Too thin ({length} < {min_thick})")
                continue
            else:
                print(f"    ✓ PASSES thickness filter")
            
            # Get std values for this segment
            seg_std_values = std_values[start:start+length]
            print(f"    Std in segment - min: {seg_std_values.min():.2f}, "
                  f"max: {seg_std_values.max():.2f}, mean: {seg_std_values.mean():.2f}")
            
            # Check what's before and after
            before_idx = max(0, start - 5)
            after_idx = min(len(std_values) - 1, start + length + 5)
            
            before_std = std_values[before_idx:start].mean() if start > 0 else 0
            after_std = std_values[start+length:after_idx].mean() if start+length < len(std_values) else 0
            
            print(f"    Context - Before (5px): {before_std:.2f}, After (5px): {after_std:.2f}")
            
            # Sample the actual image content
            if is_horizontal:
                segment_slice = gray_image[start:start+length, :]
                before_slice = gray_image[max(0, start-10):start, :] if start > 0 else None
                after_slice = gray_image[start+length:min(gray_image.shape[0], start+length+10), :]
            else:
                segment_slice = gray_image[:, start:start+length]
                before_slice = gray_image[:, max(0, start-10):start] if start > 0 else None
                after_slice = gray_image[:, start+length:min(gray_image.shape[1], start+length+10)]
            
            seg_mean_intensity = segment_slice.mean()
            before_mean = before_slice.mean() if before_slice is not None and before_slice.size > 0 else 0
            after_mean = after_slice.mean() if after_slice.size > 0 else 0
            
            print(f"    Intensity - Segment: {seg_mean_intensity:.1f}, "
                  f"Before: {before_mean:.1f}, After: {after_mean:.1f}")
            
            # Determine if this looks like a real separator
            is_bright = seg_mean_intensity > 200
            has_content_before = before_std > 15 and before_mean < 200
            has_content_after = after_std > 15 and after_mean < 200
            
            print(f"    Analysis:")
            print(f"      - Is bright/white? {is_bright} (mean={seg_mean_intensity:.1f})")
            print(f"      - Has content before? {has_content_before} (std={before_std:.1f})")
            print(f"      - Has content after? {has_content_after} (std={after_std:.1f})")
            
            if is_bright and has_content_before and has_content_after:
                print(f"    ✓✓ LIKELY VALID SEPARATOR")
            elif not is_bright:
                print(f"      WARNING: Not very bright - might be cutting through text!")
            elif not (has_content_before and has_content_after):
                print(f"     WARNING: Might be at edge or in empty area")

    def _save_image(self, img, filename):
        """Save image to file"""
        cv2.imwrite(filename, img)
        print(f"  Saved: {filename}")

    def _plot_std_distribution(self, row_std, col_std, save_prefix):
        """Plot standard deviation distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Row std histogram
        axes[0, 0].hist(row_std, bins=50, edgecolor='black')
        axes[0, 0].set_title('Row Std Distribution')
        axes[0, 0].set_xlabel('Standard Deviation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=np.percentile(row_std, 20), color='r', linestyle='--', 
                          label=f'20th percentile: {np.percentile(row_std, 20):.1f}')
        axes[0, 0].legend()
        
        # Col std histogram
        axes[0, 1].hist(col_std, bins=50, edgecolor='black')
        axes[0, 1].set_title('Column Std Distribution')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=np.percentile(col_std, 20), color='r', linestyle='--',
                          label=f'20th percentile: {np.percentile(col_std, 20):.1f}')
        axes[0, 1].legend()
        
        # Row std line plot
        axes[1, 0].plot(row_std)
        axes[1, 0].set_title('Row Std Values (y-axis position)')
        axes[1, 0].set_xlabel('Row index')
        axes[1, 0].set_ylabel('Std')
        axes[1, 0].axhline(y=20, color='r', linestyle='--', label='Threshold=20')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Col std line plot
        axes[1, 1].plot(col_std)
        axes[1, 1].set_title('Column Std Values (x-axis position)')
        axes[1, 1].set_xlabel('Column index')
        axes[1, 1].set_ylabel('Std')
        axes[1, 1].axhline(y=20, color='r', linestyle='--', label='Threshold=20')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_04_std_analysis.png', dpi=150)
        plt.close()
        print(f"  Saved: {save_prefix}_04_std_analysis.png")

    def _visualize_masks(self, gray, h_mask, v_mask, r_thresh, c_thresh, save_prefix):
        """Visualize where masks identify 'empty' areas"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Original grayscale
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Grayscale')
        axes[0].axis('off')
        
        # Horizontal mask overlay
        h_mask_img = np.zeros_like(gray)
        for i, is_empty in enumerate(h_mask):
            if is_empty:
                h_mask_img[i, :] = 255
        
        overlay1 = cv2.addWeighted(gray, 0.7, h_mask_img.astype(np.uint8), 0.3, 0)
        axes[1].imshow(overlay1, cmap='gray')
        axes[1].set_title(f'Horizontal "Empty" Rows (red)\nThreshold: {r_thresh:.1f}')
        axes[1].axis('off')
        
        # Vertical mask overlay
        v_mask_img = np.zeros_like(gray)
        for i, is_empty in enumerate(v_mask):
            if is_empty:
                v_mask_img[:, i] = 255
        
        overlay2 = cv2.addWeighted(gray, 0.7, v_mask_img.astype(np.uint8), 0.3, 0)
        axes[2].imshow(overlay2, cmap='gray')
        axes[2].set_title(f'Vertical "Empty" Columns (red)\nThreshold: {c_thresh:.1f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_05_masks.png', dpi=150)
        plt.close()
        print(f"  Saved: {save_prefix}_05_masks.png")

    def _visualize_separators(self, roi_bgr, h_segments, v_segments, 
                             h_min_thick, v_min_thick, bw, bh, save_prefix):
        """Draw all detected separators on the image"""
        img_copy = roi_bgr.copy()
        
        # Calculate minimum thicknesses
        min_h_thickness = max(h_min_thick, int(0.02 * bh))
        min_v_thickness = max(v_min_thick, int(0.02 * bw))
        
        # Draw horizontal separators
        for seg in h_segments:
            start = seg['start']
            length = seg['length']
            if length >= min_h_thickness:
                color = (0, 255, 0)  # Green for valid
                cv2.rectangle(img_copy, (0, start), (bw, start + length), color, 2)
                cv2.putText(img_copy, f"H:{length}px", (5, start + length//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (0, 165, 255)  # Orange for rejected
                cv2.rectangle(img_copy, (0, start), (bw, start + length), color, 1)
        
        # Draw vertical separators
        for seg in v_segments:
            start = seg['start']
            length = seg['length']
            if length >= min_v_thickness:
                color = (255, 0, 0)  # Blue for valid
                cv2.rectangle(img_copy, (start, 0), (start + length, bh), color, 2)
                cv2.putText(img_copy, f"V:{length}px", (start + 2, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (255, 165, 0)  # Cyan for rejected
                cv2.rectangle(img_copy, (start, 0), (start + length, bh), color, 1)
        
        cv2.imwrite(f'{save_prefix}_06_separators_overlay.png', img_copy)
        print(f"  Saved: {save_prefix}_06_separators_overlay.png")

    def _segment_boolean_mask(self, mask: np.ndarray) -> List[dict]:
        """Segment consecutive True values"""
        segments = []
        current_start = None

        for i, val in enumerate(mask):
            if val and current_start is None:
                current_start = i
            elif not val and current_start is not None:
                segments.append({'start': current_start, 'length': i - current_start})
                current_start = None

        if current_start is not None:
            segments.append({'start': current_start, 'length': len(mask) - current_start})

        return segments


if __name__ == "__main__":
    image_path = "/Users/eshagore/Desktop/wildlab-research/vizweb-python/distinct_images/ARG_593517b3943af.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Format: (x, y, width, height)
    block_bounds = (0, 0, image.shape[1], image.shape[0])  # Full image
    # block_bounds = (100, 100, 500, 300)
    
    # Create debugger and run
    debugger = DebugSpaceSeparatorExtractor(base_std_threshold=10)
    debugger.debug_extract(
        block_bounds, 
        image,
        h_min_thickness=15,
        v_min_thickness=15,
        r_threshold=20,
        c_threshold=20,
        save_prefix="debug_output"
    )
    
    print("\nDebugging complete! Check the generated images and console output.")