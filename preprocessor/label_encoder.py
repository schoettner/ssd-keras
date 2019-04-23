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
                 feature_map_sizes: np.ndarray = np.array([[38, 38],
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
                                                ]),
                 iou_threshold: float = 0.5):
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.feature_map_sizes = feature_map_sizes
        self.ratios = ratios
        self.iou_threshold = iou_threshold

        # basic calculations which are required for all label conversions
        self.amount_of_feature_maps = len(feature_map_sizes)
        self.num_bboxes_per_layer = self.__calculate_num_default_boxes_per_scale__(ratios)
        self.class_predictions = np.identity(self.num_classes)

        self.default_boxes = []
        for k, feature_map in enumerate(feature_map_sizes):
            s_k, s_k_alt = self.calculate_feature_map_scale(k, self.amount_of_feature_maps)
            self.default_boxes.append(
                self.calculate_default_boxes_for_scale(feature_map_width=feature_map[0],
                                                       feature_map_height=feature_map[1],
                                                       aspect_ratios=self.ratios[k],
                                                       s_k=s_k,
                                                       s_k_alt=s_k_alt))
        self.default_box_vector = np.concatenate(self.default_boxes, axis=0)

    def convert_label(self, ground_truth_labels: list):
        """
        main method to compute the label
        :param ground_truth_labels: list of ground truth boxes.
        the format is [class_id, x_center ,y_center, w, h]
        :return: y_true for a single label (not a batch)
        format [scale][x_cell][y_cell][box_nr][x_diff, y_diff, w_diff, h_diff, class_1, ..., class_n]
        """

        y_true = []
        for feature_map_number in range(self.amount_of_feature_maps):
            y_true.append(self.create_scale(feature_map_number))

        if not ground_truth_labels:
            return y_true  # return empty labels

        for box in ground_truth_labels:
            iou = self.calculate_iou(box[1:])
            matches = np.argwhere(iou > self.iou_threshold)
            for match in matches:
                scale, x_cell, y_cell, box_nr = self.__convert_index__(match)
                match_id = int(match)
                geo_diff = self.__calculate_geometry_difference(box[1:],
                                                                self.default_box_vector[match_id])
                y_true[scale][x_cell][y_cell][box_nr][0:4] = geo_diff
                y_true[scale][x_cell][y_cell][box_nr][4:] = self.class_predictions[box[0]]

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

    def calculate_iou(self, true_box: np.ndarray):
        """
        todo: vectorize this.
        compute the intersection of union (jaccard overlap) of the ground truth boxe
        with the default boxes (or anchor boxes)
        :param true_box: array[int] - [x,y,w,h] as centroids
        :return:
        """

        iou = np.zeros(len(self.default_box_vector))
        for idx, default_box in enumerate(self.default_box_vector):
            iou[idx] = self.calculate_box_iou(true_box, default_box)
        return iou

    def calculate_default_boxes_for_scale(self,
                                          feature_map_width: int,
                                          feature_map_height: int,
                                          aspect_ratios: np.ndarray,
                                          s_k: float,
                                          s_k_alt: float,
                                          offset: float = 0.5):
        """
        calculate the absolute coordinates of the default boxes (known as anchor box in yolo)
        this helps to compute the the jaccard overlap (iou) of a ground truth box with matching default boxes

        :param feature_map_width: int - amount of cells in x
        :param feature_map_height: int - amount of cells in y
        :param aspect_ratios: list[float] - ratios of this scale
        :param s_k: float(0...1) - scale factor for this feature map. layers with many cells have a smaller scale
        :param s_k_alt: float(0...1) - s'_k is an alternative scale factor if the layer also contains the aspect ratio 1
        :param offset: float(0...1) - offset of the bounding box center. 0.5 is the default and results in centroids
        :return:
        """
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
        calculate s_k and s'_k for the given feature map index
        :param k: number of scale [0,m-1]. origin in paper is [1,m] but was simplified here
        :param m: number of feature maps for prediction. default are 6 scales
        :return: s_k, s'_k
        """
        s_k = self.__calculate_scale__(k, m)
        s_k_next = 1 if (k == m - 1) else self.__calculate_scale__(k + 1, m)
        s_k_alt = np.sqrt(s_k * s_k_next)
        return s_k, s_k_alt

    @staticmethod
    def calculate_box_iou(a: np.ndarray, b: np.ndarray):
        """
        calculate the iou of two boxes
        :param a: [x,y,w,h] as centroid
        :param b: [x,y,w,h] as centroid
        :return: float of the iou
        """
        x = 0
        y = 1
        w = 2
        h = 3

        # calculate intersection
        x1 = np.max([a[x] - 0.5 * a[w], b[x] - 0.5 * b[w]])
        x2 = np.min([a[x] + 0.5 * a[w], b[x] + 0.5 * b[w]])
        y1 = np.max([a[y] - 0.5 * a[h], b[y] - 0.5 * b[h]])
        y2 = np.min([a[y] + 0.5 * a[h], b[y] + 0.5 * b[h]])
        intersection_area = max((x2 - x1), 0) * max((y2 - y1), 0)

        # calculate box area
        a_area = a[w] * a[h]
        b_area = b[w] * b[h]

        # calculate iou
        iou = intersection_area / (a_area + b_area - intersection_area)
        return iou  # avoid rounding issues

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

    @staticmethod
    def __convert_index__(index: int):
        scale = 1
        x_cell = 0
        y_cell = 0
        box_nr = 0
        return scale, x_cell, y_cell, box_nr

    @staticmethod
    def __calculate_geometry_difference(a: np.ndarray, b: np.ndarray):
        x_diff = 0
        y_diff = 0
        w_diff = 0
        h_diff = 0
        return np.array([x_diff, y_diff, w_diff, h_diff])
