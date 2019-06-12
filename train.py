import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_local
import pickle

train_chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]


def read_training_data(train_dir):
    features, labels = [], []
    for t_char in train_chars:
        char_images = os.listdir('{}\\{}'.format(train_dir, t_char))
        for img in char_images:

            img = imread('{}\\{}\\{}'.format(train_dir, t_char, img), as_gray=True)

            binary_image = img < threshold_local(img, block_size=51)
            features.append(binary_image.reshape(-1))
            labels.append(t_char)

    return np.array(features), np.array(labels)


def cross_validation(model, num_of_fold, train_data, train_label):

    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print(accuracy_result)


train_dir = './images/train/chars2'
images, labels = read_training_data(train_dir)

model = SVC(kernel='linear')

cross_validation(model, 2, images, labels)

model.fit(images, labels)

filename = './model2.pkl'
pickle.dump(model, open(filename, 'wb'))
