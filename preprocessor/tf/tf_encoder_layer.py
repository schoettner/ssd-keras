import tensorflow as tf
from tensorflow import Tensor


# import tensorflow.keras.backend as K


class EncoderLayer(tf.keras.layers.Layer):
    """https://www.tensorflow.org/alpha/guide/keras/custom_layers_and_models

    """

    def __init__(self):
        super(EncoderLayer, self).__init__()

        # provided values
        self.image_width = tf.Variable(initial_value=300,
                                       name='img_width',
                                       dtype=tf.float32,
                                       trainable=False)
        self.image_height = tf.Variable(initial_value=300,
                                        name='img_height',
                                        dtype=tf.float32,
                                        trainable=False)
        self.num_classes = tf.Variable(initial_value=2,
                                       name='num_classes',
                                       dtype=tf.float32,
                                       trainable=False)
        self.feature_map_size = tf.Variable(initial_value=[38, 38],
                                            name='feature_map_size',
                                            dtype=tf.float32,
                                            trainable=False)
        self.ratios = tf.Variable(initial_value=[1.0, 2.0, 3.0, 0.5, 1.0 / 3.0],
                                  name='ratios',
                                  dtype=tf.float32,
                                  trainable=False)

        self.scaling_factor = tf.Variable(initial_value=0.9,
                                          name='s_k',
                                          dtype=tf.float32,
                                          trainable=False)

        self.scaling_factor_plus_1 = tf.Variable(initial_value=0.89,
                                                 name='s_k_plus_1',
                                                 dtype=tf.float32,
                                                 trainable=False)

        self.iou_threshold = tf.Variable(initial_value=0.5,
                                         name='iou_threshold',
                                         dtype=tf.float32,
                                         trainable=False)

        self.number_boxes = tf.Variable(initial_value=6,
                                        name='number_boxes',
                                        dtype=tf.float32,
                                        trainable=False)

        self.class_predictions = tf.Variable(initial_value=tf.eye(2),
                                             name='class_identity_matrix',
                                             dtype=tf.float32,
                                             trainable=False)

    def build(self, input_shape):
        """
        enable lazy init of layer. build is executed on first __call__()
        """

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({})
        # required for the layer to be serializable
        return config

    def call(self, ground_truth: Tensor, **kwargs) -> tuple:
        return (tf.zeros(shape=[1, 3]), tf.zeros(shape=[1, 3]))

    def call_random(self) -> tuple:
        scale1 = tf.random_uniform(shape=[38, 38, 4, 6], dtype=tf.float32)
        scale2 = tf.random_uniform(shape=[19, 19, 6, 6], dtype=tf.float32)
        scale3 = tf.random_uniform(shape=[10, 10, 6, 6], dtype=tf.float32)
        scale4 = tf.random_uniform(shape=[5, 5, 6, 6], dtype=tf.float32)
        scale5 = tf.random_uniform(shape=[3, 3, 4, 6], dtype=tf.float32)
        scale6 = tf.random_uniform(shape=[1, 1, 4, 6], dtype=tf.float32)
        encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
        return encoded_label
