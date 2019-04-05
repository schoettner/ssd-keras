import os
import random
from PIL import Image
import numpy as np


class BatchLoader(object):
    """
    load a batch of x, y
    """

    def __init__(self, files: list, batch_size: int = 8,
                 apply_shuffle: bool = True,
                 validate_data: bool = False):
        self.apply_shuffle = apply_shuffle
        self.batch_size = batch_size
        self.files = files
        if validate_data:
            self.validate_files()

        total_image_count = len(self.files)  # total amount of input files
        self.last_possible_index = total_image_count - 1
        self.last_full_batch_index = total_image_count - total_image_count % batch_size - 1
        self.file_index = 0  # keep track of the current position in the file index

    def validate_files(self):
        """
        check if all image files have a corresponding label file or even exits
        this requires that the images are named with a *.jpg format!
        """
        for image in self.files:
            if not os.path.exists(image):
                self.files.remove(image)
                continue
            label_file = image[:-3] + 'txt'
            if not os.path.exists(label_file):
                self.files.remove(image)

    def load_next_batch(self):
        """
        load a batch of images and their labels
        :return:
        """
        if self.file_index == 0 and self.apply_shuffle:
            random.shuffle(self.files)
        x = []  # empty list for all the images
        y = []  # empty list for all the labels
        for _ in range(self.batch_size):
            img_file = self.files[self.file_index]
            label_file = img_file[:-3] + 'txt'
            x.append(self.load_image(img_file))  # add image to batch
            y.append(self.load_label(label_file))  # add label to batch
            self.file_index += 1  # move the pointer forward

        # check if the next for loop has enough files for a full batch
        if self.file_index == self.last_full_batch_index or self.file_index == self.last_possible_index:
            self.file_index = 0  # reset the set
            # todo shuffle the list new. this is kind of a bad spot because it delays the return
            # thats the end for today. thanks for joining
        return x, y


    @staticmethod
    def load_image(img: str, img_width: int = 300, img_height: int = 300):
        """
        load the image file. resize it to the desired resolution and put into correct input format (float32) for the model
        as to be a RGB image
        :param img: the path to the image file
        :param img_width: destination width of the output image
        :param img_height: destination height of the output image
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
