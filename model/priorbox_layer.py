from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.layers.base import Layer
import tensorflow.keras.backend as K


class PriorBox(Layer):
    """
    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output.
    """

    def __init__(self, num_bboxes, **kwargs):
        self.num_bboxes = num_bboxes
        super(PriorBox, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(PriorBox, self).build(input_shape)

    def call(self, x, mask=None):
        # todo
        output = K.l2_normalize(x, self.axis)
        output *= self.scale
        return output

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        # the last dimension is (x, y, w, h, ∆x, ∆y, ∆w, ∆h)
        return (batch_size, feature_map_height, feature_map_width, self.num_bboxes, 8)


    def get_config(self):
        config = {
            'num_bboxes': self.num_bboxes
        }
        base_config = super(PriorBox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))