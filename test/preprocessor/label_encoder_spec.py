import numpy as np
import pytest

from preprocessor.label_encoder import LabelEncoder


class LabelEncoderSpec:

    def test_scale_creation(self):
        label_encoder = self.given_default_encoder()

        # create all the default scales
        scale1 = label_encoder.create_scale(0)
        scale2 = label_encoder.create_scale(1)
        scale3 = label_encoder.create_scale(2)
        scale4 = label_encoder.create_scale(3)
        scale5 = label_encoder.create_scale(4)
        scale6 = label_encoder.create_scale(5)

        # check values for scale 1. for the others check only the shape
        assert scale1.shape == (38, 38, 6, 7)
        assert scale1.dtype == np.float32
        assert scale1.min() == 0
        assert scale1.max() == 0
        assert scale2.shape == (19, 19, 6, 7)
        assert scale3.shape == (10, 10, 6, 7)
        assert scale4.shape == (5, 5, 6, 7)
        assert scale5.shape == (3, 3, 6, 7)
        assert scale6.shape == (1, 1, 6, 7)

    def test_y_true_size(self):
        label_encoder = self.given_default_encoder()
        y_true = label_encoder.convert_label(None)

        # only check that all scales are returned, not the shape of the scales
        assert len(y_true) == 6

    def test_feature_map_scale(self):
        label_encoder = self.given_default_encoder()

        #scale 1
        s1, s2 = label_encoder.calculate_feature_map_scale(0, 6)
        assert s1 == 0.2
        assert s2 == pytest.approx(0.26, abs=0.01)
        #scale 2
        s1, s2 = label_encoder.calculate_feature_map_scale(1, 6)
        assert s1 == pytest.approx(0.34)
        assert s2 == pytest.approx(0.4, abs=0.01)
        #scale 3
        s1, s2 = label_encoder.calculate_feature_map_scale(2, 6)
        assert s1 == 0.48
        assert s2 == pytest.approx(0.55, abs=0.01)
        #scale 4
        s1, s2 = label_encoder.calculate_feature_map_scale(3, 6)
        assert s1 == pytest.approx(0.62)
        assert s2 == pytest.approx(0.69, abs=0.01)
        #scale 5
        s1, s2 = label_encoder.calculate_feature_map_scale(4, 6)
        assert s1 == 0.76
        assert s2 == pytest.approx(0.82, abs=0.01)
        #scale 6
        s1, s2 = label_encoder.calculate_feature_map_scale(5, 6)
        assert s1 == pytest.approx(0.9)
        assert s2 == pytest.approx(0.95, abs=0.01)

    def test_calculate_boxes_for_layer(self):
        label_encoder = self.given_default_encoder()
        ratios = self.given_default_ratios()

        # biggest layer
        default_boxes = label_encoder.calculate_boxes_for_layer(feature_map_width=1,
                                                                feature_map_height=1,
                                                                aspect_ratios=ratios,
                                                                s_k=0.9,
                                                                s_k_alt=0.95)
        expected_boxes = np.array([
            [150, 150, 270, 270],
            [150, 150, 381.84, 190.92],
            [150, 150, 467.65, 155.88],
            [150, 150, 190.92, 381.84],
            [150, 150, 155.88, 467.65],
            [150, 150, 285, 285],
        ])

        np.testing.assert_allclose(default_boxes, expected_boxes, atol=0.1)

    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)

    @staticmethod
    def given_default_ratios():
        return np.array([1, 2, 3, 1/2, 1/3], dtype=np.float32)
