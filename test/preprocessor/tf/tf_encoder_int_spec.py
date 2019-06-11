import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf.tf_encoder_layer import EncoderLayer


class TfEncoderIntSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_small_feature_map(self):
        # with tf.Session() as sess:
        ground_truth = tf.constant(([0, 74, 73, 111, 110],
                                    [0, 15, 15, 7, 7]), dtype=tf.int32)
        encoder = self.given_small_map_encoder()
        label = encoder.call(ground_truth)
        print('label shape: {}'.format(label.get_shape()))
        print('label: {}'.format(label))
        assert True

    @staticmethod
    def given_small_map_encoder() -> EncoderLayer:
        return EncoderLayer(feature_map_size=[2, 2],
                            ratios=[1, 2, 3, 1 / 2, 1 / 3],
                            s_k=0.76,
                            s_k_alt=0.82,
                            num_boxes=5, )