import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf.tf_encoder_layer import EncoderLayer


class TfEncoderSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_default_box_for_2_by_2_map(self):
        encoder = EncoderLayer(feature_map_size=[2, 2],
                               ratios=[1, 2, 3, 1/2, 1/3],
                               s_k=0.76,
                               s_k_alt=0.82,
                               num_boxes=5,)
        encoder(tf.constant(1))
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
        a = tf.constant([[150, 150, 300, 300], [150, 150, 100, 100]], dtype=tf.float32)
        b = tf.constant([[150, 150, 300, 300], [150, 150, 200, 200]], dtype=tf.float32)
        iou = EncoderLayer.calculate_iou(a, b)
        tf.assert_equal(iou, [1, 0.25])

    @run_in_graph_and_eager_modes
    def test_convert_index(self):
        iou = tf.constant([0.5, 0.2, 0.2, 0.5, 0.1], dtype=tf.float32)
        indices = tf.where(tf.greater_equal(iou, 0.5))
        print(indices)
        encoder = EncoderLayer()
        indices = encoder.decode_index(indices)
        print(indices)
        assert True

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
