import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf.tf_encoder_layer import EncoderLayer

class TfEncoderSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_label_convert(self):
        encoder = EncoderLayer()
        label = encoder(tf.constant(1))
        default_boxes = encoder.default_boxes
        print(default_boxes)
        assert True

    @run_in_graph_and_eager_modes
    def test_cartesian_calculation(self):

        a = tf.constant((1, 2))
        b = tf.constant((3, 4))
        c = EncoderLayer.cartesian_product(a,b)
        print(c)
        cartesian = ([[1, 2], [1, 4]],
                     [[2, 3], [2, 4]])
        # tf.assert_equal(c, cartesian)
