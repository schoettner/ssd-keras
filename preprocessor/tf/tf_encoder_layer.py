import tensorflow as tf
from tensorflow import Tensor

import tensorflow.keras.backend as K

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
                 num_boxes: int = 6):
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

        self.default_boxes = self.__calculate_default_boxes()

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
        center_y = tf.linspace(start=offset + cell_pixel_height,
                               stop=(offset + cells_on_y - 1) * cell_pixel_height,
                               num=cells_on_y)
        cartesian = self.cartesian_product(center_x, center_y)
        num_cells = cells_on_x * cells_on_y
        # for each cell. add the boxes to the center
        # e.g. box at 75,75 with boxes [[114,114][162,80][80,162]]
        # will be
        grid = K.repeat(cartesian, num_boxes_per_cell)
        w_h = tf.tile(boxes_w_h, (num_cells, 1))
        print(w_h)
        # asdf = tf.concat((grid, boxes_w_h))
        print('grid after repeat')
        print(grid)
        #[75, 75, 114, 114]
        #[75, 75, 162, 80]
        #[75, 75, 80, 162]
        return cartesian

    @staticmethod
    def cartesian_product(a: Tensor, b: Tensor) -> Tensor:
        # https://stackoverflow.com/questions/47132665/cartesian-product-in-tensorflow
        c = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
        c = tf.reshape(c, (-1, 2))
        return c

    def build(self, input_shape):
        """
        enable lazy init of layer. build is executed on first __call__()
        """
        print('build the default boxes')


    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({})
        # required for the layer to be serializable
        return config

    def call(self, ground_truth: Tensor, **kwargs) -> Tensor:
        print('call')
        # shape = feature map size, boxes of layer, 4 + num classes
        label = tf.zeros(shape=self.label_output_shape, name='encoded_label')
        print(label.get_shape())
        return label

    def call_random(self) -> tuple:
        scale1 = tf.random_uniform(shape=[38, 38, 4, 6], dtype=tf.float32)
        scale2 = tf.random_uniform(shape=[19, 19, 6, 6], dtype=tf.float32)
        scale3 = tf.random_uniform(shape=[10, 10, 6, 6], dtype=tf.float32)
        scale4 = tf.random_uniform(shape=[5, 5, 6, 6], dtype=tf.float32)
        scale5 = tf.random_uniform(shape=[3, 3, 4, 6], dtype=tf.float32)
        scale6 = tf.random_uniform(shape=[1, 1, 4, 6], dtype=tf.float32)
        encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
        return encoded_label
