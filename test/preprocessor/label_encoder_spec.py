import numpy as np

from preprocessor.batch_loader import BatchLoader


class LabelEncoderSpec:

    def test_image_resize(self):
        batch_loader = self.given_default_batch_loader()
        image = batch_loader.load_image('test/resources/test_picture.jpg')
        assert image.shape == (300, 300)
        assert image.dtype == np.float32

    def test_load_label(self):
        batch_loader = self.given_default_batch_loader()

        # label with one box
        label = batch_loader.load_label('test/resources/test_picture.txt')
        expected_label = np.array([
            [0, 1, 2, 3, 4]
        ])
        assert label.shape == (1, 5)
        np.testing.assert_equal(label, expected_label)

        # label with two boxes
        label = batch_loader.load_label('test/resources/test_picture2.txt')
        expected_label = np.array([
            [0, 1, 2, 3, 4],
            [1, 5, 6, 7, 8]
        ])
        assert label.shape == (2, 5)
        np.testing.assert_equal(label, expected_label)

        # label with three boxes
        label = batch_loader.load_label('test/resources/test_picture3.txt')
        expected_label = np.array([
            [0, 1, 2, 3, 4],
            [1, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
        ])
        assert label.shape == (3, 5)
        np.testing.assert_equal(label, expected_label)



    @staticmethod
    def given_default_batch_loader():
        return BatchLoader(None)
