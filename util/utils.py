
from tensorflow.keras.utils import plot_model


def print_model(model, filename='model.png'):
    plot_model(model, to_file=filename)
