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
        self.ratios = ratios
        self.num_bboxes_per_layer = self.calculate_num_boxes_per_layer(ratios)

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

    def calculate_jaccard_overlap(self, true_boxes: np.ndarray):
        """
        simple IOU (intersection of union) calculation
        iou = (Area of overlap) / (Area of union)
        calculate for whole scale at once? all scales at once?
        call incoming. brb

        :return:
        """


        return 0

    def get_box_label(self):
        return None

    def set_true_boxes(self):
        return None


    @staticmethod
    def calculate_num_boxes_per_layer(ratios: np.ndarray):
        num_bboxes_per_layer = []
        for layer in ratios:
            if 1 in layer:
                # add s´k if 1 is included (ssd paper page 6)
                num_bboxes_per_layer.append(len(layer) + 1)
            else:
                num_bboxes_per_layer.append(len(layer))
        return num_bboxes_per_layer
