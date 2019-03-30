from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.layers.base import Layer
import tensorflow.keras.backend as K


class Priorbox(Layer):

    def __init__(self, num_bboxes, **kwargs):
        self.num_bboxes = num_bboxes
        super(Priorbox, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(Priorbox, self).build(input_shape)

    def call(self, x, mask=None):
        # todo
        output = K.l2_normalize(x, self.axis)
        output *= self.scale
        return output

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.num_bboxes, 8)


    def get_config(self):
        config = {
            'num_bboxes': self.num_bboxes
        }
        base_config = super(Priorbox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))