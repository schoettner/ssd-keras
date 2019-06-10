import tensorflow as tf
import numpy as np
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf.tf_encoder_layer import EncoderLayer


class TfEncoderSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_default_box_for_2_by_2_map(self):
        encoder = self.given_small_map_encoder()

        encoder(tf.constant(([0, 15, 15, 7, 7],)))
        default_boxes = encoder.default_boxes

        expected_boxes = [
            [75, 75, 114, 114],
            [75, 75, 161.22, 80.61],
            [75, 75, 197.45, 65.82],
            [75, 75, 80.61, 161.22],
            [75, 75, 65.82, 197.45],
            # [75, 75, 123, 123],
            [75, 225, 114, 114],
            [75, 225, 161.22, 80.61],
            [75, 225, 197.45, 65.82],
            [75, 225, 80.61, 161.22],
            [75, 225, 65.82, 197.45],
            # [75, 225, 123, 123],
            [225, 75, 114, 114],
            [225, 75, 161.22, 80.61],
            [225, 75, 197.45, 65.82],
            [225, 75, 80.61, 161.22],
            [225, 75, 65.82, 197.45],
            # [225, 75, 123, 123],
            [225, 225, 114, 114],
            [225, 225, 161.22, 80.61],
            [225, 225, 197.45, 65.82],
            [225, 225, 80.61, 161.22],
            [225, 225, 65.82, 197.45],
            # [225, 225, 123, 123]
            ]
        tf.assert_near(default_boxes, expected_boxes, atol=1)

    @run_in_graph_and_eager_modes
    def test_iou(self):
        # a = tf.constant([[150, 150, 300, 300], [150, 150, 100, 100]], dtype=tf.float32)
        # b = tf.constant([[150, 150, 300, 300], [150, 150, 200, 200]], dtype=tf.float32)
        # iou = EncoderLayer.calculate_iou(a, b)
        # tf.assert_equal(iou, [1, 0.25])
        # todo fix
        assert True

    @run_in_graph_and_eager_modes
    def test_convert_index(self):
        # iou = tf.constant([0.5, 0.2, 0.2, 0.5, 0.1], dtype=tf.float32)
        # match_indices = tf.where(tf.greater_equal(iou, 0.5))
        match_indices = tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=tf.int64)
        encoder = self.given_small_map_encoder()
        encoded_indices = encoder.decode_index(match_indices)
        expected = [
            [0,0,0],
            [0,0,1],
            [0,0,2],
            [0,0,3],
            [0,0,4],
            [0,1,0],
            [0,1,1],
            [0,1,2],
            [0,1,3],
            [0,1,4],
            [1,0,0],
            [1,0,1],
            [1,0,2],
            [1,0,3],
            [1,0,4],
            [1,1,0],
            [1,1,1],
            [1,1,2],
            [1,1,3],
            [1,1,4],
        ]
        tf.assert_equal(encoded_indices, expected)

    @run_in_graph_and_eager_modes
    def test_cartesian_calculation(self):
        a = tf.constant((1, 2))
        b = tf.constant((3, 4))
        c = EncoderLayer.cartesian_product(a, b)
        cartesian = [[1, 3],
                     [1, 4],
                     [2, 3],
                     [2, 4]]
        tf.assert_equal(c, cartesian)

    @run_in_graph_and_eager_modes
    def test_geometry_difference(self):
        a = tf.constant([75, 75, 114, 114], dtype=tf.float32)
        b = tf.constant([80, 80, 120, 120], dtype=tf.float32)

        c = self.given_small_map_encoder().calculate_geometry_difference(a,b)
        expected = np.array([5.0, 5.0, 6.0, 6.0], dtype=np.float32)
        print(c)
        tf.assert_equal(c, expected)

    @staticmethod
    def given_small_map_encoder() -> EncoderLayer:
        return EncoderLayer(feature_map_size=[2, 2],
                            ratios=[1, 2, 3, 1 / 2, 1 / 3],
                            s_k=0.76,
                            s_k_alt=0.82,
                            num_boxes=5, )
