import os

from skimage.io import imread
from skimage.transform import resize

from detection import detect_plate
from segmentation import segment_chars
from reading import read

dir = 'C:\\projects\\car-plates\\images\\demo'
pics = os.listdir(dir)

for pic in pics:
    car_image = imread('{}\\{}'.format(dir, pic), as_gray=True)
    car_image = resize(car_image, (480, 640), anti_aliasing=True)
    car_plate = detect_plate(car_image)
    chars = segment_chars(car_plate)
    read(chars)
