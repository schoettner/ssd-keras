import tensorflow as tf

class Loss:

    def __init__(self):
        print('loss function for SSD')

    def calculate_loss(self, y_true, y_pred):
        return tf.keras.backend.sum(y_pred)