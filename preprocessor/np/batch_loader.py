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
        self.last_possible_index = total_image_count - 1  # last valid index to check if you run through the list
        self.last_full_batch_index = total_image_count - total_image_count % batch_size - 1  # last index hat has a full batch
        self.file_index = 0  # keep track of the current position in the file index

    def validate_files(self):
        """
        check if all image files have a corresponding label file or even exits
        this requires that the images are named with a *.jpg (RGB) format! and the label with *.txt format
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
            resized_image, input_width, input_height = self.load_image(img_file)
            x.append(resized_image)  # add image to batch
            y.append(self.load_label(label_file, origin_width=input_width, origin_height=input_height))  # add label to batch
            self.file_index += 1  # move the pointer forward

        # check if the next for loop has enough files for a full batch
        if self.file_index == self.last_full_batch_index or self.file_index == self.last_possible_index:
            self.file_index = 0  # reset the files for a new epoch
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
        width, height = img.size
        img = img.resize(size=(img_width, img_height))
        return np.array(img, dtype=np.float32), width, height

    @staticmethod
    def load_label(label: str,
                   origin_width: int, origin_height: int,
                   destination_width: int = 300, destination_height: int = 300):
        """
        load the label text file with the bounding box details. one file can contain multiple boxes
        the coordinates are also mapped to the resized image
        input format in file class_1, x_min_1, y_min_1, x_max_1, y_max_1, class_2, ...
        the label will be processed by the encoder later
        :param label: path to the text file
        :param origin_width: width of input image
        :param origin_height: height of input image
        :param destination_width: width of resized image
        :param destination_height: height of resized image
        :return: numpy array of the labels in shape=(num_boxes, 5) and dtpye=int32. the format is [class,x,y,w,h]
                 or None if the file could not be loaded
        """
        # compute the coordination scale factors
        width_scale = destination_width / origin_width
        height_scale = destination_height / origin_height
        # load label in a row vector and then convert to matrix with 5 cols
        label = np.loadtxt(label, delimiter=',')
        reshaped_label = label.reshape((-1, 5))
        # convert bounding box to resized image
        reshaped_label[:, [1, 3]] *= width_scale
        reshaped_label[:, [2, 4]] *= height_scale
        # convert x_min, y_min, x_max, y_max to x,y,w,h
        x = (reshaped_label[:, 1] + reshaped_label[:, 3]) // 2  # x = (x_min + x_max) / 2
        y = (reshaped_label[:, 2] + reshaped_label[:, 4]) // 2  # y = (y_min + y_max) / 2
        w = (reshaped_label[:, 3] - reshaped_label[:, 1])  # y = (x_max - x_min)
        h = (reshaped_label[:, 4] - reshaped_label[:, 2])  # y = (y_max - y_min)
        reshaped_label[:, 1] = x
        reshaped_label[:, 2] = y
        reshaped_label[:, 3] = w
        reshaped_label[:, 4] = h
        return reshaped_label.astype(np.int32)
