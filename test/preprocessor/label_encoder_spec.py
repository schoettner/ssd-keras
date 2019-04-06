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

        assert scale1.shape == (38, 38, 6, 7)
        assert scale2.shape == (19, 19, 6, 7)
        assert scale3.shape == (10, 10, 6, 7)
        assert scale4.shape == (5, 5, 6, 7)
        assert scale5.shape == (3, 3, 6, 7)
        assert scale6.shape == (1, 1, 6, 7)


    @staticmethod
    def given_default_encoder():
        return LabelEncoder(num_classes=3)
