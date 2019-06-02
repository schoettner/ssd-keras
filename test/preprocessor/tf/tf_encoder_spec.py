import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf.tf_encoder_layer import EncoderLayer

class TfEncoderSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_label_convert(self):
        encoder = EncoderLayer()
        label = encoder(tf.constant(1))
        test = encoder.default_boxes
        assert True