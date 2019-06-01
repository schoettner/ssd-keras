import tensorflow as tf
from tensorflow import Tensor

def encode_label(ground_truth: Tensor) -> tuple:
    raise NotImplementedError('not yet implemented for tensorflow')


def random_label(ground_truth: Tensor) -> tuple:
    # return random data for the moment. will be fixed later
    scale1 = tf.random_uniform(shape=[38, 38, 4, 6], dtype=tf.float32)
    scale2 = tf.random_uniform(shape=[19, 19, 6, 6], dtype=tf.float32)
    scale3 = tf.random_uniform(shape=[10, 10, 6, 6], dtype=tf.float32)
    scale4 = tf.random_uniform(shape=[5, 5, 6, 6], dtype=tf.float32)
    scale5 = tf.random_uniform(shape=[3, 3, 4, 6], dtype=tf.float32)
    scale6 = tf.random_uniform(shape=[1, 1, 4, 6], dtype=tf.float32)
    encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
    return encoded_label
