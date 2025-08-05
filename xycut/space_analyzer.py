import numpy as np

# This function finds the tightest content region inside the given bounds by trimming whitespace margins
# from top, bottom, left, and right. It mimics the behavior of the Java SpaceAnalyzer.findContentBounds.
def find_content_bounds(image, bounds):
    x, y, w, h = bounds
    sub_image = image[y:y+h, x:x+w]

    # Convert to grayscale if the image is in RGB
    if len(sub_image.shape) == 3:
        #todo: convert to grayscale
        sub_image = np.mean(sub_image, axis=2)

    # Define threshold: pixels below this are considered content
    threshold = 250
    content_mask = sub_image < threshold # todo: double check scalar vs array comparison

    # Check for any non-background pixels in each row and column
    rows_with_content = np.any(content_mask, axis=1)
    cols_with_content = np.any(content_mask, axis=0)

    # If no content found, return original bounds
    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return bounds

    # Find trimmed boundaries
    top = np.argmax(rows_with_content)
    bottom = len(rows_with_content) - 1 - np.argmax(rows_with_content[::-1])
    left = np.argmax(cols_with_content)
    right = len(cols_with_content) - 1 - np.argmax(cols_with_content[::-1])

    new_x = x + left
    new_y = y + top
    new_w = right - left + 1
    new_h = bottom - top + 1

    return (new_x, new_y, new_w, new_h)
