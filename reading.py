import pickle

from skimage.transform import resize


def read(characters):
    if characters == (None, None):
        return

    model = pickle.load(open('./model2.pkl', 'rb'))

    plate_string = ''

    for char, _ in characters:
        char = resize(char, (20, 20))
        char = char.reshape(1, -1)
        plate_string += model.predict(char)[0]

    print(plate_string)
    a = 1
