from skimage.filters import threshold_otsu, threshold_local
from components import plate_component, show_components


def to_grayscale(image):
    """
    Helper method, converts RGB images into grayscale
    :param image: RGB image
    :return: Grayscale image
    """
    return image * 255


def to_binary(gray_image):
    """
    Converts grayscale image to binary image using local threshold
    :param gray_image: Input image converted to grayscale
    :return: binary image
    """
    threshold_value = threshold_local(gray_image,block_size=51, offset=10)
    bin_image = gray_image > threshold_value
    return bin_image


def detect_plate(image):
    """
    Runner method for license plate component detection. Takes the input image and calls
    helper methods to convert it to binary. Then calls methods to find connected components on the image and
    identify if there is a car license plate present on the image
    :param image: RGB image
    :return: Cropped binary image. Bounding box of the license plate if it is present on the
    input image, False otherwise
    """
    gray_image = to_grayscale(image)
    bin_image = to_binary(gray_image)
    show_components(gray_image, bin_image)
    plate = plate_component(bin_image)
    return plate
