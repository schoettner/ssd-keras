import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import Tensor


class EncoderLayer(tf.keras.layers.Layer):
    """https://www.tensorflow.org/alpha/guide/keras/custom_layers_and_models

    """

    def __init__(self,
                 img_width: int = 300,
                 img_height: int = 300,
                 num_classes: int = 2,
                 feature_map_size: [] = (38, 38),
                 ratios: [] = (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0),
                 s_k: float = 0.9,
                 s_k_alt: float = 0.89,
                 iou: float = 0.5,
                 num_boxes: int = 5):
        super(EncoderLayer, self).__init__()

        # provided values
        self.image_width = tf.constant(img_width,
                                       name='const_img_width',
                                       dtype=tf.float32)
        self.image_height = tf.constant(img_height,
                                        name='const_img_height',
                                        dtype=tf.float32)
        self.num_classes = tf.constant(num_classes,
                                       name='const_num_classes',
                                       dtype=tf.float32)
        self.feature_map_size = tf.constant(feature_map_size,
                                            name='const_feature_map_size',
                                            dtype=tf.float32)
        self.ratios = tf.constant(ratios,
                                  name='const_ratios',
                                  dtype=tf.float32)

        self.scaling_factor = tf.constant(s_k,
                                          name='const_s_k',
                                          dtype=tf.float32)

        self.scaling_factor_plus_1 = tf.constant(s_k_alt,
                                                 name='const_s_k_plus_1',
                                                 dtype=tf.float32)

        self.iou_threshold = tf.constant(iou,
                                         name='const_iou_threshold',
                                         dtype=tf.float32)

        self.number_boxes = tf.constant(num_boxes,
                                        name='const_number_boxes',
                                        dtype=tf.float32)

        self.class_predictions = tf.eye(num_classes, name='const_class_identity_matrix')
        self.label_output_shape = (*feature_map_size, num_boxes, num_classes + 4)
        self.num_boxes_on_layer = feature_map_size[0] * feature_map_size[1], num_boxes

        self.default_boxes = self.__calculate_default_boxes(cells_on_x=feature_map_size[0],
                                                            cells_on_y=feature_map_size[1],
                                                            img_width=img_width,
                                                            img_height=img_height,
                                                            num_boxes_per_cell=num_boxes)

    def build(self, input_shape):
        """
        enable lazy init of layer. build is executed on first __call__()
        """
        print('build was called')

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({})
        # required for the layer to be serializable
        return config

    def call(self, ground_truth: Tensor, **kwargs) -> Tensor:
        # shape = feature map size, boxes of layer, 4 + num classes

        label = tf.zeros(shape=self.label_output_shape, name='encoded_label')
        if ground_truth is None:
            return label

        num_boxes_on_layer = self.label_output_shape[0] * self.label_output_shape[1] * self.label_output_shape[2]
        ground_truth_box = ground_truth[:, 1:]
        repeated_box = K.repeat(ground_truth_box, num_boxes_on_layer)
        repeated_ground_truth = tf.cast(repeated_box, dtype=tf.float32)
        iou = self.calculate_iou(self.default_boxes[:, :, ], repeated_ground_truth)
        matches = tf.where(tf.greater_equal(iou, self.iou_threshold))
        indices = self.decode_index(matches)
        self.set_values(indices, matches, ground_truth, label)
        return label

    @staticmethod
    def call_random() -> tuple:
        scale1 = tf.random_uniform(shape=[38, 38, 4, 6], dtype=tf.float32)
        scale2 = tf.random_uniform(shape=[19, 19, 6, 6], dtype=tf.float32)
        scale3 = tf.random_uniform(shape=[10, 10, 6, 6], dtype=tf.float32)
        scale4 = tf.random_uniform(shape=[5, 5, 6, 6], dtype=tf.float32)
        scale5 = tf.random_uniform(shape=[3, 3, 4, 6], dtype=tf.float32)
        scale6 = tf.random_uniform(shape=[1, 1, 4, 6], dtype=tf.float32)
        encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
        return encoded_label

    def __calculate_default_boxes(self,
                                  cells_on_x: int = 38,
                                  cells_on_y: int = 38,
                                  img_width: int = 300,
                                  img_height: int = 300,
                                  num_boxes_per_cell: int = 6,
                                  offset: float = 0.5):

        cell_pixel_width = img_width / cells_on_x
        cell_pixel_height = img_height / cells_on_y
        ratios_sqrt = tf.sqrt(self.ratios)

        # calculate the absolute width and height of the default boxes
        # see SSD paper page 6 for calculation details
        box_widths = self.scaling_factor * cell_pixel_width * ratios_sqrt
        box_heights = self.scaling_factor * cell_pixel_height / ratios_sqrt
        # if 1 in self.ratios:  # does not work
        #     tf.concat(box_width, self.scaling_factor_plus_1 * cell_pixel_width)
        #     tf.concat(box_height, self.scaling_factor_plus_1 * cell_pixel_height)

        # calculate x and y center of the cells
        boxes_w_h = tf.stack((box_widths, box_heights), axis=1)
        center_x = tf.linspace(start=offset * cell_pixel_width,
                               stop=(offset + cells_on_x - 1) * cell_pixel_width,
                               num=cells_on_x)
        center_y = tf.linspace(start=offset * cell_pixel_height,
                               stop=(offset + cells_on_y - 1) * cell_pixel_height,
                               num=cells_on_y)
        cartesian_center = self.cartesian_product(center_x, center_y)
        num_cells = cells_on_x * cells_on_y
        center_grid = K.repeat(cartesian_center, num_boxes_per_cell)
        center_grid_full = tf.reshape(center_grid, shape=(num_cells * num_boxes_per_cell, 2))
        w_h = tf.tile(boxes_w_h, (num_cells, 1))
        default_boxes = tf.concat([center_grid_full, w_h], axis=1)
        return default_boxes

    def set_values(self,
                   indices_label: Tensor,
                   indices_default_box: Tensor,
                   ground_truth: Tensor,
                   label: Tensor):
        """ set the geometry and class value everywhere where the indices match
        """
        # todo
        # print('label: {}'.format(label))
        # print('ground truth: {}'.format(ground_truth))
        index_label = indices_label[0]
        index_label_flat = tf.reshape(index_label, shape=[3])  # index in label as flat vector (for concat + sparse)

        # print('index in label: {}'.format(index_label))
        # index_default_box = indices_default_box[0]
        # print('default_box_index: {}'.format(index_default_box))

        # create a sparse tensor with the box prediction
        box_geometry = ground_truth[0][1:]  # get the geometry data from ground truth
        box_geometry = tf.cast(box_geometry, dtype=tf.float32)
        print("box_geometry: {}".format(box_geometry))
        index_default_box = indices_default_box[0]  # get the indices of the best matching default boxes
        print("default_box_index: {}".format(index_default_box))
        default_box = self.default_boxes[0]
        print("default_box_geometry: {}".format(default_box))

        geo_diff = self.calculate_geometry_difference(box_geometry, default_box)
        x_index = tf.constant(0, dtype=tf.int32, shape=[1])
        y_index = tf.constant(1, dtype=tf.int32, shape=[1])
        w_index = tf.constant(2, dtype=tf.int32, shape=[1])
        h_index = tf.constant(3, dtype=tf.int32, shape=[1])
        label_x_index = tf.concat([index_label_flat, x_index], axis=0)
        label_y_index = tf.concat([index_label_flat, y_index], axis=0)
        label_w_index = tf.concat([index_label_flat, w_index], axis=0)
        label_h_index = tf.concat([index_label_flat, h_index], axis=0)
        sparse_box_label = tf.SparseTensor(indices=[label_x_index,
                                                    label_y_index,
                                                    label_w_index,
                                                    label_h_index],  # create a sparse tensor that sets the value
                                             values=geo_diff,
                                             dense_shape=self.label_output_shape)
        print('box label: {}'.format(tf.sparse.to_dense(sparse_box_label)))

        # create a sparse tensor with the class prediction
        class_id = ground_truth[0][0]  # get the class id from the ground truth box
        class_index = tf.constant(4, dtype=tf.int32, shape=[1]) + class_id  # get the index of the correct class in label
        label_class_index = tf.concat([index_label_flat, class_index], axis=0)  # concat to the absolute class coordinate
        sparse_class_label = tf.SparseTensor(indices=[label_class_index],  # create a sparse tensor that sets the value
                                             values=[1],  # set the classification for this class to 100%
                                             dense_shape=self.label_output_shape)
        print('class: {}'.format(tf.sparse.to_dense(sparse_class_label)))

    def decode_index(self, a: Tensor) -> Tensor:
        """
        decode the index. converts an index to the direct coordinate in the scale
        [cell_x][cell_y][default_box][geo_offset, one-hot label]
        the decoded the indices that
        """
        assert a.dtype == tf.int64, 'tensor is not of type int64, can not convert'

        num_boxes_tensor = tf.cast(self.number_boxes, dtype=a.dtype)
        # index of the cell from 0..n required to calc x and y position of index
        cell_index = tf.floor_div(a, num_boxes_tensor)

        cells_on_x = tf.cast(self.feature_map_size[0], dtype=a.dtype)
        x = tf.floor_div(cell_index, cells_on_x)

        cells_on_y = tf.cast(self.feature_map_size[1], dtype=a.dtype)
        y = tf.mod(cell_index, cells_on_y)

        boxes = tf.mod(a, num_boxes_tensor)

        stacked = tf.stack((x, y, boxes), axis=1)  # concat to (-1, 3)
        return tf.cast(stacked, dtype=tf.int32)

    @staticmethod
    def calculate_geometry_difference(ground_truth: Tensor, default_box: Tensor) -> Tensor:
        return tf.subtract(default_box, ground_truth)

    @staticmethod
    def calculate_iou(a: Tensor, b: Tensor) -> Tensor:
        # todo fix this
        b = b[0]  # just take one box from the tensor for the moment
        x = 0
        y = 1
        w = 2
        h = 3

        # transfer to x_min, x_max coordinates
        x1 = tf.maximum(a[:, x] - 0.5 * a[:, w], b[:, x] - 0.5 * b[:, w])
        x2 = tf.minimum(a[:, x] + 0.5 * a[:, w], b[:, x] + 0.5 * b[:, w])
        y1 = tf.maximum(a[:, y] - 0.5 * a[:, h], b[:, y] - 0.5 * b[:, h])
        y2 = tf.minimum(a[:, y] + 0.5 * a[:, h], b[:, y] + 0.5 * b[:, h])
        intersection_area = tf.maximum((x2 - x1), 0) * tf.maximum((y2 - y1), 0)

        # calculate box area
        a_area = a[:, w] * a[:, h]
        b_area = b[:, w] * b[:, h]

        # calculate iou
        iou = intersection_area / (a_area + b_area - intersection_area)
        return iou

    @staticmethod
    def cartesian_product(a: Tensor, b: Tensor) -> Tensor:
        # https://stackoverflow.com/questions/47132665/cartesian-product-in-tensorflow
        c = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
        c = tf.reshape(c, (-1, 2))
        return c