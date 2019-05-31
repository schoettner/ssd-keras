import numpy as np

from preprocessor.np.batch_loader import BatchLoader


class BatchLoaderSpec:

    def test_file_validation(self):
        batch_loader = self.given_default_batch_loader()

        # label file for image exists
        batch_loader.validate_files()
        assert batch_loader.files == ['test/resources/test_picture.jpg']

        # label file for image does not exits
        batch_loader.files = ['test/resources/no_label.jpg']
        batch_loader.validate_files()
        assert batch_loader.files == []

        # image file itself does not exit
        batch_loader.files = ['test/resources/test_picture2.jpg']
        batch_loader.validate_files()
        assert batch_loader.files == []

    def test_image_resize(self):
        batch_loader = self.given_default_batch_loader()
        image, width, height = batch_loader.load_image('test/resources/test_picture.jpg')
        assert image.shape == (300, 300, 3)
        assert image.dtype == np.float32
        assert width == 1024
        assert height == 768

    def test_load_label(self):
        batch_loader = self.given_default_batch_loader()

        # label with one box
        label = batch_loader.load_label('test/resources/test_picture.txt', origin_width=300, origin_height=300)
        expected_label = np.array([
            [0, 4, 5, 4, 6]
        ])
        assert label.shape == (1, 5)
        np.testing.assert_equal(label, expected_label)

        # label with two boxes
        label = batch_loader.load_label('test/resources/test_picture2.txt', origin_width=300, origin_height=300)
        expected_label = np.array([
            [0, 4, 5, 4, 6],
            [1, 5, 4, 6, 4]
        ])
        assert label.shape == (2, 5)
        np.testing.assert_equal(label, expected_label)

        # label with three boxes
        label = batch_loader.load_label('test/resources/test_picture3.txt', origin_width=300, origin_height=300)
        expected_label = np.array([
            [0, 4, 5, 4, 6],
            [1, 5, 4, 6, 4],
            [0, 2, 2, 3, 3],
        ])
        assert label.shape == (3, 5)
        np.testing.assert_equal(label, expected_label)

    def test_load_label_with_resize(self):
        batch_loader = self.given_default_batch_loader()

        # label with one box
        label = batch_loader.load_label('test/resources/test_720p_image.txt', origin_width=1024, origin_height=768)
        expected_label = np.array([
            [0, 146, 97, 292, 195]
        ])
        assert label.shape == (1, 5)
        np.testing.assert_equal(label, expected_label)

    @staticmethod
    def given_default_batch_loader():
        return BatchLoader(files=['test/resources/test_picture.jpg'])
