from PIL import Image
import numpy as np


class BatchLoader(object):
    """
    load a batch of x, y
    """

    def __init__(self, files: list, batch_size: int = 8, apply_shuffle: bool = True):
        self.apply_shuffle = apply_shuffle
        self.batch_size = batch_size
        self.files = files

    @staticmethod
    def load_image(img: str, img_width: int = 300, img_height: int = 300):
        """
        load the image file. resize it to the desired resolution and put into correct input format (float32) for the model
        :param img: the path to the image file
        :return: numpy array of the image in shape=(width, height, 3) and dtype=float32
                 or None if the file could not be loaded
        """
        img = Image.open(img)
        img = img.resize(size=(img_width, img_height))
        return np.array(img, dtype=np.float32)

    @staticmethod
    def load_label(label: str):
        """
        load the label text file with the bounding box details. one file can contain multiple boxes
        input format in file class_1, x_min_1, y_min_1, x_max_1, y_max_1, class_2, ...
        the label will be processed by the encoder later
        :param label: path to the text file
        :return: numpy array of the labels in shape=(num_boxes, 5) and dtpye=int32
                 or None if the file could not be loaded
        """
        label = np.loadtxt(label, delimiter=',', dtype=np.int32)  # load label in a row vector
        return label.reshape((-1, 5))  # reshape to a row matrix with 5 cols
