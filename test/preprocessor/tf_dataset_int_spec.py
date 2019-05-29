from preprocessor import tf_dataset
import tensorflow as tf

from preprocessor.Params import Params


class TfDatasetIntegrationSpec:

    def test_full_converter(self):
        with tf.Session() as sess:
            train_filenames = self.given_test_image()
            train_labels = self.given_test_labels()
            params = self.given_test_params()
            train_inputs = tf_dataset.input_fn(True, train_filenames, train_labels, params)
            sess.run([train_inputs.initializer])
            result = train_inputs.get_next()
            sess.close()

        assert True

    @staticmethod
    def given_test_image() -> []:
        return ['test/resources/no_label.jpg', 'test/resources/test_picture.jpg']

    @staticmethod
    def given_test_labels() -> []:
        return ['test/resources/test_picture.txt', 'test/resources/test_picture2.txt']

    @staticmethod
    def given_test_params() -> Params:
        return Params('test/resources/params.json')