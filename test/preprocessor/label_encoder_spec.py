import numpy as np

from preprocessor.label_encoder import LabelEncoder


class LabelEncoderSpec:

    def test_y_true_size(self):
        label_encoder = self.given_default_encoder()
        y_true = label_encoder.convert_label(None)

        # only check that all scales are returned, not the shape of the scales
        assert len(y_true) == 6

    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)

    @staticmethod
    def given_default_ratios():
        return np.array([1, 2, 3, 1/2, 1/3], dtype=np.float32)
