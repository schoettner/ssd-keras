import tensorflow as tf
from tensorflow import Tensor


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(EncoderLayer, self).__init__()

    def call(self, ground_truth: Tensor, **kwargs) -> tuple:
        return (tf.zeros(shape=[1, 3]), tf.zeros(shape=[1, 3]))

    def call_random(self) -> tuple:
        scale1 = tf.random_uniform(shape=[38, 38, 4, 6], dtype=tf.float32)
        scale2 = tf.random_uniform(shape=[19, 19, 6, 6], dtype=tf.float32)
        scale3 = tf.random_uniform(shape=[10, 10, 6, 6], dtype=tf.float32)
        scale4 = tf.random_uniform(shape=[5, 5, 6, 6], dtype=tf.float32)
        scale5 = tf.random_uniform(shape=[3, 3, 4, 6], dtype=tf.float32)
        scale6 = tf.random_uniform(shape=[1, 1, 4, 6], dtype=tf.float32)
        encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
        return encoded_label
