import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy


class Loss:
    """
    L(x, c, l, g) = 1/N * (L_conf(x, c) + α * L_loc(x, l, g))
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = tf.constant(alpha, name='loc_loss_weight')

    def calculate_loss(self, y_true, y_pred):
        """
        ignore the 1/N since it is a linear factor that only requires calc time without purpose
        """
        l_loc = self._calculate_smooth_L1_loss(y_true[:, :4], y_pred[:, :4])  # first 4 values
        l_conf = self._calculate_softmax_loss(y_true[:, 4:], y_pred[:, 4:])  # starting with the 5th to the end
        return l_conf + self.alpha * l_loc

    @staticmethod
    def _calculate_smooth_L1_loss(y_true, y_pred):
        """
        calculate the l1 smooth loss from faster R-CNN https://arxiv.org/abs/1504.08083
        :param y_true: ∆(x,y,w,h) from the label
        :param y_pred: ∆(x,y,w,h) from the prediction
        :return: loss as scalar
        """
        x = y_true - y_pred
        x_abs = tf.abs(x)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(x_abs, 1.0),  # if
                           square_loss,  # then
                           x_abs - 0.5)  # else
        return tf.reduce_sum(l1_loss)

    @staticmethod
    def _calculate_softmax_loss(y_true, y_pred):
        """
        use the default cross entropy from keras instead
        :param y_true: class predictions from label
        :param y_pred: class predictions from prediction
        :return: loss as scalar
        """
        return categorical_crossentropy(y_true, y_pred)
