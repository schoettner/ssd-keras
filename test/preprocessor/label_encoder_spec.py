import numpy as np

from preprocessor.label_encoder import LabelEncoder


class LabelEncoderSpec:

    # def test_y_true_size(self):
    #     label_encoder = self.given_default_encoder()
    #     y_true = label_encoder.convert_label(None)
    #
    #     # only check that all scales are returned, not the shape of the scales
    #     assert len(y_true) == 6

    def test_iou_shape(self):
        label_encoder = self.given_default_encoder()

        iou = label_encoder.calculate_iou(None)
        assert iou.shape == (11640, 4)

    def test_iou_box_calculation(self):
        label_encoder = self.given_default_encoder()

        # boxes do not overlap
        iou = label_encoder.calculate_box_iou(np.array([30, 30, 10, 10]), np.array([100, 100, 10, 10]))
        assert iou == 0

        # boxes are equal
        iou = label_encoder.calculate_box_iou(np.array([50, 50, 20, 20]), np.array([50, 50, 20, 20]))
        assert iou == 1

        # boxes cover some area
        iou = label_encoder.calculate_box_iou(np.array([75, 75, 50, 50]), np.array([75, 100, 50, 50]))
        assert iou == 1/3


    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)

    @staticmethod
    def given_default_ratios():
        return np.array([1, 2, 3, 1/2, 1/3], dtype=np.float32)
