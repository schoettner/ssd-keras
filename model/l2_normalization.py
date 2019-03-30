from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.layers.base import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


class L2Normalization(Layer):
    """
    Since Keras does not provide a L2Normalization Layer, this is added.
    It learns its scaling factor as trainable parameter.
    The SSD paper suggests to init the scaling with 20 (see section 3.1 PASCAL VOC2007)

    Arguments:
        scale_init: The initial scaling parameter. Default is 20

    Input shape:
        4D tensor of shape (batch, width, height, channels)

    Returns:
        The scaled tensor. Same shape as the input tensor

    References:
          https://github.com/lvaleriu/ssd_keras-1/blob/master/keras_layer_L2Normalization.py
          http://cs.unc.edu/~wliu/papers/parsenet.pdf
    """

    def __init__(self, scale_init=20, **kwargs):
        self.scale_init = scale_init
        self.axis = 3  # normalize of the channel
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.scale_init * tf.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
