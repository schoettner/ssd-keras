import numpy as np
import pytest

from preprocessor.label_encoder import LabelEncoder


class LabelEncoderInitSpec:
    """
    test everything that happens during initialization of the encoder
    """

    def test_zero_scale_creation(self):
        label_encoder = self.given_default_encoder()

        # check values for scale 1. for the others check only the shape
        scale1 = label_encoder.create_scale(0)
        assert scale1.shape == (38, 38, 6, 7)
        assert scale1.dtype == np.float32
        assert scale1.min() == 0
        assert scale1.max() == 0

        scale2 = label_encoder.create_scale(1)
        assert scale2.shape == (19, 19, 6, 7)

        scale3 = label_encoder.create_scale(2)
        assert scale3.shape == (10, 10, 6, 7)

        scale4 = label_encoder.create_scale(3)
        assert scale4.shape == (5, 5, 6, 7)

        scale5 = label_encoder.create_scale(4)
        assert scale5.shape == (3, 3, 6, 7)

        scale6 = label_encoder.create_scale(5)
        assert scale6.shape == (1, 1, 6, 7)


    def test_feature_map_scale_calculation(self):
        label_encoder = self.given_default_encoder()

        # scale 1
        s1, s2 = label_encoder.calculate_feature_map_scale(0, 6)
        assert s1 == 0.2
        assert s2 == pytest.approx(0.26, abs=0.01)

        # scale 2
        s1, s2 = label_encoder.calculate_feature_map_scale(1, 6)
        assert s1 == pytest.approx(0.34)
        assert s2 == pytest.approx(0.4, abs=0.01)

        # scale 3
        s1, s2 = label_encoder.calculate_feature_map_scale(2, 6)
        assert s1 == 0.48
        assert s2 == pytest.approx(0.55, abs=0.01)

        # scale 4
        s1, s2 = label_encoder.calculate_feature_map_scale(3, 6)
        assert s1 == pytest.approx(0.62)
        assert s2 == pytest.approx(0.69, abs=0.01)

        # scale 5
        s1, s2 = label_encoder.calculate_feature_map_scale(4, 6)
        assert s1 == 0.76
        assert s2 == pytest.approx(0.82, abs=0.01)

        # scale 6
        s1, s2 = label_encoder.calculate_feature_map_scale(5, 6)
        assert s1 == pytest.approx(0.9)
        assert s2 == pytest.approx(0.95, abs=0.01)

    def test_calculate_boxes_for_biggest_layer(self):
        label_encoder = self.given_default_encoder()
        ratios = self.given_default_ratios()

        # biggest layer with [1][1]
        default_boxes = label_encoder.calculate_default_boxes_for_scale(feature_map_width=1,
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

    def test_calculate_boxes_for_second_biggest_layer(self):
        label_encoder = self.given_default_encoder()
        ratios = self.given_default_ratios()
        # theoretical [2][2] layer to keep the amount of values small enough
        default_boxes = label_encoder.calculate_default_boxes_for_scale(feature_map_width=2,
                                                                        feature_map_height=2,
                                                                        aspect_ratios=ratios,
                                                                        s_k=0.76,
                                                                        s_k_alt=0.82)
        expected_boxes = np.array([
            [75, 75, 114, 114],
            [75, 75, 161.22, 80.61],
            [75, 75, 197.45, 65.82],
            [75, 75, 80.61, 161.22],
            [75, 75, 65.82, 197.45],
            [75, 75, 123, 123],
            [75, 225, 114, 114],
            [75, 225, 161.22, 80.61],
            [75, 225, 197.45, 65.82],
            [75, 225, 80.61, 161.22],
            [75, 225, 65.82, 197.45],
            [75, 225, 123, 123],
            [225, 75, 114, 114],
            [225, 75, 161.22, 80.61],
            [225, 75, 197.45, 65.82],
            [225, 75, 80.61, 161.22],
            [225, 75, 65.82, 197.45],
            [225, 75, 123, 123],
            [225, 225, 114, 114],
            [225, 225, 161.22, 80.61],
            [225, 225, 197.45, 65.82],
            [225, 225, 80.61, 161.22],
            [225, 225, 65.82, 197.45],
            [225, 225, 123, 123],

        ])
        np.testing.assert_allclose(default_boxes, expected_boxes, atol=0.1)

    def test_class_prediction_matrix(self):
        label_encoder = self.given_default_encoder()
        expected = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        actual = label_encoder.class_predictions
        np.testing.assert_equal(actual, expected)

    def test_default_boxes_shape(self):
        label_encoder = self.given_default_encoder()
        assert label_encoder.default_boxes[0].shape == (8664, 4)  # 6x38x38
        assert label_encoder.default_boxes[1].shape == (2166, 4)  # 6x19x19
        assert label_encoder.default_boxes[2].shape == (600, 4)  # 6x10x10
        assert label_encoder.default_boxes[3].shape == (150, 4)  # 6x5x5
        assert label_encoder.default_boxes[4].shape == (54, 4)  # 6x3x3
        assert label_encoder.default_boxes[5].shape == (6, 4)  # 6x1x1

    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)

    @staticmethod
    def given_default_ratios():
        return np.array([1, 2, 3, 1/2, 1/3], dtype=np.float32)
