import os
import tensorflow as tf

def initialize_tensorflow():
    # Enable detailed TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    # Check TensorFlow and CUDA versions
    print("TensorFlow Version:", tf.__version__)

    # Check available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")

    # Configure TensorFlow to use all available GPUs
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Ensure CUDA_VISIBLE_DEVICES is correctly set
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', 'Not Set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

# Initialize TensorFlow
initialize_tensorflow()

# Main script code
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
from typing import List, Tuple
from scipy.ndimage import label

def plot_it(image_array: np.array) -> None:
    """
    This function shows the plot of a given image (np.array)
    Args:
        image_array (np.array): Image
    """
    plt.rcParams["figure.figsize"] = (20, 15)
    _, ax = plt.subplots(1)
    ax.imshow(image_array)
    plt.show()

def binarize_image(image_array: np.array, threshold="otsu") -> np.array:
    """
    This function takes a np.array that represents an RGB image and returns
    the binarized image (np.array) by applying the otsu threshold.
    Args:
        image_array (np.array): image
        threshold (str, optional): "otsu" or a float. Defaults to "otsu".
    Returns:
        np.array: binarized image
    """
    grayscale = rgb2gray(image_array)
    if threshold == "otsu":
        threshold = threshold_otsu(grayscale)
    binarized_image_array = grayscale > threshold
    return binarized_image_array

def get_seeds(image_array: np.array, mask_array: np.array, exclusion_mask: np.array) -> List[Tuple[int, int]]:
    """
    This function takes an array that represents an image and a mask.
    It returns a list of tuples with indices of seeds in the structure
    covered by the mask.
    The seed pixels are defined as pixels in the inner 80% of the mask which
    are not white in the image.
    Args:
        image_array (np.array): Image
        mask_array (np.array): Mask array of shape (y, x)
        exclusion_mask (np.array): Exclusion mask
    Returns:
        List[Tuple[int, int]]: [(x,y), (x,y), ...]
    """
    mask_y_values, mask_x_values = np.where(mask_array)
    # Define boundaries of the inner 80% of the mask
    mask_y_diff = mask_y_values.max() - mask_y_values.min()
    mask_x_diff = mask_x_values.max() - mask_x_values.min()
    x_min_limit = mask_x_values.min() + mask_x_diff / 10
    x_max_limit = mask_x_values.max() - mask_x_diff / 10
    y_min_limit = mask_y_values.min() + mask_y_diff / 10
    y_max_limit = mask_y_values.max() - mask_y_diff / 10
    # Define intersection of mask and image
    mask_coordinates = set(zip(mask_y_values, mask_x_values))
    image_y_values, image_x_values = np.where(np.invert(image_array))
    image_coordinates = set(zip(image_y_values, image_x_values))

    intersection_coordinates = mask_coordinates & image_coordinates
    # Select intersection coordinates that are in the inner 80% of the mask
    seed_pixels = []
    for y_coord, x_coord in intersection_coordinates:
        if x_coord < x_min_limit:
            continue
        if x_coord > x_max_limit:
            continue
        if y_coord < y_min_limit:
            continue
        if y_coord > y_max_limit:
            continue
        if exclusion_mask[y_coord, x_coord]:
            continue
        seed_pixels.append((x_coord, y_coord))
    return seed_pixels

def detect_horizontal_and_vertical_lines(image: np.ndarray, max_depiction_size: Tuple[int, int]) -> np.ndarray:
    """
    This function takes an image and returns a binary mask that labels the pixels that
    are part of long horizontal or vertical lines.
    Args:
        image (np.ndarray): binarised image (np.array; type bool) as it is returned by
            binary_erosion() in complete_structure_mask()
        max_depiction_size (Tuple[int, int]): height, width; used as thresholds
    Returns:
        np.ndarray: Exclusion mask that contains indices of pixels that are part of
            horizontal or vertical lines
    """
    binarised_im = ~image * 255
    binarised_im = binarised_im.astype("uint8")

    structure_height, structure_width = max_depiction_size

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structure_width, 1))
    horizontal_mask = cv2.morphologyEx(
        binarised_im, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    horizontal_mask = horizontal_mask == 255

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structure_height))
    vertical_mask = cv2.morphologyEx(
        binarised_im, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    vertical_mask = vertical_mask == 255

    return horizontal_mask + vertical_mask

def find_equidistant_points(x1: int, y1: int, x2: int, y2: int, num_points: int = 5) -> List[Tuple[int, int]]:
    """
    Finds equidistant points between two points.
    Args:
        x1 (int): x coordinate of first point
        y1 (int): y coordinate of first point
        x2 (int): x coordinate of second point
        y2 (int): y coordinate of second point
        num_points (int, optional): Number of points to return. Defaults to 5.
    Returns:
        List[Tuple[int, int]]: Equidistant points on the given line
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        points.append((x, y))
    return points

def detect_lines(image: np.ndarray, max_depiction_size: Tuple[int, int], segmentation_mask: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns a binary mask that labels the pixels that
    are part of lines that are not part of chemical structures (like arrays, tables).
    Args:
        image (np.ndarray): binarised image (np.array; type bool) as it is returned by
            binary_erosion() in complete_structure_mask()
        max_depiction_size (Tuple[int, int]): height, width; used for thresholds
        segmentation_mask (np.ndarray): Indicates whether or not a pixel is part of a
            chemical structure depiction (shape: (height, width))
    Returns:
        np.ndarray: Exclusion mask that contains indices of pixels that are part of
            horizontal or vertical lines
    """
    image = ~image * 255
    image = image.astype("uint8")
    # Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(image,
                            1,
                            np.pi / 180,
                            threshold=5,
                            minLineLength=int(max(max_depiction_size)/4),
                            maxLineGap=10)
    # Generate exclusion mask based on detected lines
    exclusion_mask = np.zeros_like(image)
    if lines is None:
        return exclusion_mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if any of the lines is in a chemical structure depiction
        points = find_equidistant_points(x1, y1, x2, y2, num_points=7)
        points_in_structure = False
        for x, y in points[1:-1]:
            if segmentation_mask[int(y), int(x)]:
                points_in_structure = True
                break
        if points_in_structure:
            continue
        cv2.line(exclusion_mask, (x1, y1), (x2, y2), 255, 3)
    return exclusion_mask.astype(bool)

def complete_structure_mask(image: np.ndarray) -> np.ndarray:
    """
    This function returns a binary mask that labels the pixels that are part of a
    chemical structure depiction.
    Args:
        image (np.ndarray): binary image (np.array; type bool)
    Returns:
        np.ndarray: structure mask
    """
    structure_mask = binary_dilation(
        binary_erosion(image, selem=np.ones((5, 5))),
        selem=np.ones((5, 5)),
    )
    return structure_mask

def detect_chemical_structures(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects chemical structures in an image.
    Args:
        image (np.ndarray): input image
    Returns:
        Tuple[np.ndarray, np.ndarray]: chemical structures mask, exclusion mask
    """
    binarized_image = binarize_image(image)
    structure_mask = complete_structure_mask(binarized_image)
    max_depiction_size = (image.shape[0] // 10, image.shape[1] // 10)
    exclusion_mask = detect_lines(~structure_mask, max_depiction_size, structure_mask)
    return structure_mask, exclusion_mask

# Example usage of the script
if __name__ == "__main__":
    # Load your image
    # image = cv2.imread('path_to_your_image.jpg')
    # structure_mask, exclusion_mask = detect_chemical_structures(image)
    # plot_it(structure_mask)
    # plot_it(exclusion_mask)
    pass
