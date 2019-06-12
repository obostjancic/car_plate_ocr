
import numpy as np
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def valid_dimensions(car_plate, region):
    """
    Determines if the region has a valid dimensions. Works based on an
    assumption that the char component will be of certain dimensions.
    Used to filter out too small and too large components.
    :param region: part of an image that was identified as a connected component
    :return: True if the region has valid dimensions, False otherwise
    """
    car_plate_h, car_plate_w = car_plate.shape
    min_h, max_h, min_w, max_w = char_dimensions = (0.30 * car_plate_h, 0.95 * car_plate_h, 0.015 * car_plate_w, 0.25 * car_plate_w)
    min_row, min_col, max_row, max_col = region.bbox
    region_h = max_row - min_row
    region_width = max_col - min_col

    return min_h < region_h < max_h and min_w < region_width < max_w


def valid_position(car_plate, region):
    """
    Determines if the region has a valid position on the original photo. Works based
    on an assumption that the char will not be somewhere on the border of the image.
    :param region: part of an image that was identified as a connected component
    :return: True if the region has a valid position, False otherwise
    """
    car_plate_w, car_plate_h = car_plate.shape[1], car_plate.shape[0]
    min_row, min_col, max_row, max_col = region.bbox

    return 0 < min_row < max_row < car_plate_h and 0 < min_col < max_col < car_plate_w


def filter_by_height(characters):
    """
    Throws out any identified components that do not have a similar height as the rest of them.
    Used to filter out non-char components on the license plates.
    :param characters: Identified character components on the car license plate
    :return: Filtered characters
    """
    avg_height = 0
    for char, col in characters:
        avg_height += char.shape[0]

    avg_height /= len(characters)

    for i in range(len(characters)-1, -1, -1):
        char = characters[i][0]
        if char.shape[0] < avg_height - 5:
            characters.pop(i)

    return characters


def extract_chars(car_plate, region, ax1):
    """
    Extracts the character from the identified region. Adapts its borders because sometimes part of chars
    get left out from the component. That creates problems later on when classifying them
    :param car_plate: Identified car license plate
    :param region: Char connected component
    :return: Adjusted char component and its X axis position which is used to sort the chars in the right order
    """
    min_row, min_col, max_row, max_col = region.bbox
    min_row -= 1
    min_col -= 1
    # max_col += 1
    # max_row += 1
    char = car_plate[min_row:max_row, min_col:max_col]

    rect_border = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rect_border)

    return char, min_col


def segment_chars(car_plate_image):
    """
    Runner method for character segmentation. Takes a binary image of the identified car license plate. Identifies
    connected components of chars on it. Calls different helpers to filter out unwanted components. Sorts the characters
    :param car_plate_image: Binary image of the car license plate
    :return: Array of binary images, bounding boxes of characters.
    """
    if car_plate_image is None:
        return None, None

    car_plate = np.invert(car_plate_image)

    labelled_plate = measure.label(car_plate)

    characters, column_list = [], []

    fig, ax1 = plt.subplots(1)
    ax1.imshow(car_plate, cmap="gray")

    for region in regionprops(labelled_plate):

        if not valid_dimensions(car_plate, region):
            continue

        if not valid_position(car_plate, region):
            continue

        char, min_col = extract_chars(car_plate, region, ax1)
        min_row, min_col, max_row, max_col = region.bbox
        characters.append((char, min_col))

        column_list.append(min_col)

    plt.show()
    characters = sorted(characters, key=lambda x: x[1])
    if characters:
        characters = filter_by_height(characters)

    return characters
