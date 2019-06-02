import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from preprocessor.tf import tf_dataset
from util.params import Params


class TfDatasetSpec(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_label_convert(self):
        img_file = tf.constant('img_name', dtype=tf.string)
        label_file = tf.constant('test/resources/test_picture2.txt', dtype=tf.string)
        _, label = tf_dataset._load_label(img_file, label_file)
        expected = [[0, 2, 2, 6, 8], [1, 2, 2, 8, 6]]
        tf.assert_equal(label, expected)

    def test_dataset_loading(self):
        with tf.Session() as sess:
            train_filenames = self.given_test_image()
            train_labels = self.given_test_labels()
            params = self.given_test_params()
            iterator, init_op = tf_dataset.input_fn(True, train_filenames, train_labels, params)
            sess.run(init_op)
            img, label = sess.run(iterator.get_next())
            assert img.shape == (2, 300, 300, 3)
            assert len(label) == 6
            assert label[0].shape == (2, 38, 38, 4, 6)
            sess.close()

    @staticmethod
    def given_test_image() -> []:
        return ['test/resources/no_label.jpg', 'test/resources/test_picture.jpg']

    @staticmethod
    def given_test_labels() -> []:
        return ['test/resources/test_picture.txt', 'test/resources/test_picture2.txt']

    @staticmethod
    def given_test_params() -> Params:
        return Params('test/resources/params.json')
