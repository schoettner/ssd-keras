import numpy as np

from preprocessor.np.label_encoder import LabelEncoder


class LabelEncoderSpec:

    def test_y_true_size(self):
        label_encoder = self.given_default_encoder()
        y_true = label_encoder.convert_label(None)

        # only check that all scales are returned, not the shape of the scales
        assert len(y_true) == 6

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

    def test_iou_vector_calculation(self):
        label_encoder = self.given_default_encoder()
        # iou = label_encoder.calculate_iou(np.array([150, 150, 80, 80]))
        iou = label_encoder.calculate_iou(np.array([150, 150, 300, 300]))
        assert len(iou) == 11640  # 8664+2166+600+150+54+6 possible boxes
        assert np.min(iou) >= 0  # if the box has w,h=300 0 is not possible
        assert np.max(iou) <= 1

    def test_simple_label_encode_single_cell(self):
        label_encoder = self.given_simple_encoder()

        # perfect match on 2x2 scale, first cell, first box
        # the other two boxes in this cell should trigger too
        box = [np.array([0, 25, 25, 27.5, 27.5])]

        y_true = label_encoder.convert_label(box)

        expected_match_box1 = np.array([0, 0, 0, 0, 1, 0, 0])  # first box in the cell has no geometry offset
        expected_match_box2 = np.array([0, 0, -9, 11, 1, 0, 0])  # second box in the cell has no geometry offset
        expected_match_box3 = np.array([0, 0, 7, 7, 1, 0, 0])  # third box in the cell has no geometry offset
        np.testing.assert_equal(y_true[1][0][0][0], expected_match_box1)
        np.testing.assert_equal(y_true[1][0][0][1], expected_match_box2)
        np.testing.assert_equal(y_true[1][0][0][2], expected_match_box3)

    def test_simple_label_encode_multi_cell(self):
        label_encoder = self.given_simple_encoder()

        # trigger three different cells in [3][3] scale
        box = [np.array([0, 15, 15, 7, 7]),
               np.array([1, 49, 51, 5, 10]),
               np.array([2, 81, 82, 12, 11]),
               ]

        y_true = label_encoder.convert_label(box)

        expected_match_box1 = np.array([1, 1, -1, -1, 1, 0, 0])  # first box in the cell has no geometry offset
        expected_match_box2 = np.array([0, -2, -1, -1, 0, 1, 0])  # second box in the cell has no geometry offset
        expected_match_box3 = np.array([1, 0, -2, -1, 0, 0, 1])  # third box in the cell has no geometry offset
        # all match in the smallest (first) scale
        np.testing.assert_equal(y_true[0][0][0][0], expected_match_box1)  # first cell, first box
        np.testing.assert_equal(y_true[0][1][1][1], expected_match_box2)  # middle cell, second box
        np.testing.assert_equal(y_true[0][2][2][2], expected_match_box3)  # last cell, third box

    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)

    @staticmethod
    def given_simple_encoder():
        return LabelEncoder(num_classes=3,
                            img_width=100,
                            img_height=100,
                            feature_map_sizes=np.array([[3, 3], [2, 2], [1, 1]]),
                            ratios=np.array([[1, 0.5], [1, 0.5], [1, 0.5]]),
                            iou_threshold=0.2)

    @staticmethod
    def given_default_ratios():
        return np.array([1, 2, 3, 1/2, 1/3], dtype=np.float32)
