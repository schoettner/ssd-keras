import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy


class Loss:
    """
    L(x, c, l, g) = 1/N * (L_conf(x, c) + α * L_loc(x, l, g))
    """

    def __init__(self):
        print('loss function for SSD')

    def calculate_loss(self, y_true, y_pred):

        # todo add the L_loc loss too
        L_conf = categorical_crossentropy(y_true[:, 4:], y_pred[:, 4:])
        return L_conf