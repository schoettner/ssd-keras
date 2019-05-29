from preprocessor import tf_dataset
from tensorflow.contrib.eager.python import tfe
import tensorflow as tf
import pytest


class TfDatasetSpec:

    @staticmethod
    @pytest.fixture(scope='session', autouse=True)
    def eager(request):
        # https://stackoverflow.com/questions/48234032/run-py-test-test-in-different-process?noredirect=1&lq=1
        print('run in eager mode')
        tfe.enable_eager_execution()

    def test_label_convert(self):
        img_file = tf.constant('img_name', dtype=tf.string)
        label_file = tf.constant('test/resources/test_picture2.txt', dtype=tf.string)
        _, label = tf_dataset._load_label(img_file, label_file)
        expected = [[0, 2, 2, 6, 8], [1, 2, 2, 8, 6]]
        tf.assert_equal(label, expected)
