import numpy as np


class LabelEncoder(object):
    """
    convert the human readable label to the input-label for the ssd model
    """

    def __init__(self,
                 num_classes: int,
                 img_width: int = 300,
                 img_height: int = 300,
                 # the default feature map sizes are taken for SSD300
                 feature_maps: np.ndarray = np.array([[38, 38],
                                                      [19, 19],
                                                      [10, 10],
                                                      [5, 5],
                                                      [3, 3],
                                                      [1, 1]]),
                 # default ratios are {1,2,3,1/2,1/3} on 6 layers in original ssd paper
                 ratios: np.ndarray = np.array([[1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                                ])):
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.feature_map_sizes = feature_maps
        self.amount_of_feature_maps = len(feature_maps)
        self.ratios = ratios
        self.num_bboxes_per_layer = self.__calculate_num_default_boxes_per_scale__(ratios)

    def convert_label(self, label: np.ndarray):
        """
        main method to compute the label
        :param label:
        :return:
        """
        y_true = []
        for feature_map_number, _ in enumerate(self.feature_map_sizes):
            y_true.append(self.create_scale(feature_map_number))
        return y_true

    def create_scale(self, feature_map_number: int, include_classes: bool = True):
        """
        create y_true for the scale. everything is initialized with zeros as float32
        :param feature_map_number: the number of the feature map [0, n]
        :param include_classes: check if the map should include the class prediction or not
        :return:
        """
        feature_map_width = self.feature_map_sizes[feature_map_number][0]
        feature_map_height = self.feature_map_sizes[feature_map_number][1]
        boxes_of_feature_map = self.num_bboxes_per_layer[feature_map_number]
        vector_size = 4 + self.num_classes if include_classes else 4

        scale_shape = (feature_map_width,
                       feature_map_height,
                       boxes_of_feature_map,
                       vector_size)
        return np.zeros(shape=scale_shape, dtype=np.float32)

    # def calculate_jaccard_overlap(self, true_boxes: np.ndarray):

    def calculate_boxes_for_layer(self,
                                  feature_map_width: int,
                                  feature_map_height: int,
                                  aspect_ratios: np.ndarray,
                                  s_k: float,
                                  s_k_alt: float,
                                  offset: float = 0.5):
        cell_width = self.img_width // feature_map_width
        cell_height = self.img_height // feature_map_height
        ar_sqrt = np.sqrt(aspect_ratios)

        # calculate absolute width and height of the default boxes
        box_width = (s_k * cell_width) * ar_sqrt
        box_height = (s_k * cell_height) / ar_sqrt
        if 1 in aspect_ratios:
            box_width = np.append(box_width, s_k_alt * cell_width)  # ar for this is 1 too
            box_height = np.append(box_height, s_k_alt * cell_height)  # ar for this is 1 too
        w_h = np.column_stack((box_width, box_height))

        # calculate the x and y center of the feature map cells
        center_x = np.linspace(start=offset * cell_width,
                               stop=(offset + feature_map_width - 1) * cell_width,
                               num=feature_map_width)
        center_y = np.linspace(start=offset * cell_height,
                               stop=(offset + feature_map_height - 1) * cell_height,
                               num=feature_map_height)

        # compute the cartesian product of x and y
        grid = self.__cartesian_product__(center_x, center_y)

        # combine to x,y,w,h (centroids)
        num_boxes_per_cell = w_h.shape[0]
        num_cells = feature_map_width*feature_map_height
        grid = np.repeat(grid, num_boxes_per_cell, axis=0)
        w_h = np.tile(w_h, (num_cells, 1))
        boxes = np.hstack((grid, w_h))
        return boxes

    def calculate_feature_map_scale(self, k: int, m: int = 6):
        """
        calculate s_k and s'_k
        :param k: number of scale [0,m-1]. origin in paper is [1,m] but was simplified here
        :param m: number of feature maps for prediction. default are 6 scales
        :return: s_k, s'_k
        """
        s_k = self.__calculate_scale__(k, m)
        s_k_next = 1 if (k == m - 1) else self.__calculate_scale__(k + 1, m)
        s_k_alt = np.sqrt(s_k * s_k_next)
        return s_k, s_k_alt

    @staticmethod
    def __cartesian_product__(a: np.ndarray, b: np.ndarray):
        """
        compute all possible combinations of two arrays, called the cartesian product

        Reference: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        :param a: first array
        :param b: second array
        :return: cartesian product of the two arrays
        """
        return np.stack(np.meshgrid(a, b), -1).reshape(-1, 2, order='F')

    @staticmethod
    def __calculate_scale__(k: int,
                            m: int = 6,
                            s_min: float = 0.2,
                            s_max: float = 0.9):
        """
        calculate the scale like described in the SSD paper (page 6)
        :param k: number of scale [0,m-1]. origin in paper is [1,m] but was simplified here
        :param m: number of feature maps for prediction. default are 6 scales
        :param s_min: min scale. default is paper 0.2
        :param s_max: max scale. default in paper is 0.9
        :return: the scale of feature map (or scale) k
        """
        return s_min + ((s_max - s_min)/(m - 1)) * k  # not k-1 because we start at 0 not 1

    @staticmethod
    def __calculate_num_default_boxes_per_scale__(ratios: np.ndarray):
        num_bboxes_per_layer = []
        for layer in ratios:
            if 1 in layer:
                # add s´k if 1 is included (ssd paper page 6)
                num_bboxes_per_layer.append(len(layer) + 1)
            else:
                num_bboxes_per_layer.append(len(layer))
        return num_bboxes_per_layer
