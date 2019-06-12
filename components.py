from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_components(gray_image, bin_image):
    """
    Shows connected components on the passed image
    :param gray_image: grayscale image
    :param bin_image: grayscale image converted to binary image
    """

    label_image = measure.label(bin_image)
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_image, cmap="gray")

    for region in regionprops(label_image):
        if region.area < 1000:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        rect_border = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

    plt.show()


def valid_w_to_h_ratio(region):
    """
    Determines if the region has a valid W to H ratio. For most license plates
    this ratio varies between 1:4 and 1:7
    :param region: part of an image that was identified as a connected component
    :return: True if the region has a valid ratio, False otherwise
    """
    min_row, min_col, max_row, max_col = region.bbox
    width, height = max_col - min_col, max_row - min_row
    ratio = width / height
    return 1 < ratio < 7


def valid_area(region):
    """
    Determines if the region has a valid area.
    :param region: part of an image that was identified as a connected component
    :return: True if the region has a valid area, False otherwise
    """
    return 4000 < region.area < 19000


def valid_position(region):
    """
    Determines if the region has a valid position on the original photo. Works based
    on an assumption that the car plate will be somewhere in the middle of the image.
    Used to filter out components that are close to the borders of the image.
    :param region: part of an image that was identified as a connected component
    :return: True if the region has a valid position, False otherwise
    """
    min_x, max_x, min_y, max_y = plate_position = (50, 590, 100, 440)
    min_row, min_col, max_row, max_col = region.bbox
    return min_row > min_y and min_col > min_x and max_row < max_y and max_col < max_x


def valid_dimensions(region):
    """
    Determines if the region has a valid dimensions. Works based on an
    assumption that the car plate component will be of certain dimensions.
    Used to filter out too small and too large components.
    :param region: part of an image that was identified as a connected component
    :return: True if the region has valid dimensions, False otherwise
    """
    min_h, max_h, min_w, max_w = plate_dimensions = (30, 160, 150, 350)
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col

    return min_h <= region_height <= max_h and min_w <= region_width <= max_w and region_width > region_height


def plate_component(bin_image):
    """
    Detects a component that resembles a car license plate on the passed image, if such exists
    :param bin_image: Input image, converted to binary image
    :return: Cropped binary image, bounding box of the car license plate if it was wound. None otherwise
    """
    label_image = measure.label(bin_image)

    for region in regionprops(label_image):

        if not valid_area(region):
            continue

        if not valid_position(region):
            continue

        if not valid_dimensions(region):
            continue

        if not valid_w_to_h_ratio(region):
            continue

        min_row, min_col, max_row, max_col = region.bbox

        return bin_image[min_row-1:max_row+1, min_col-1:max_col+1]

    return None
