import yaml
from tensorflow.keras.utils import plot_model


def load_config_file(config_file: str = 'config.yml'):
    with open(config_file, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as ex:
            print(ex)


def print_model(model, filename='model.png'):
    plot_model(model, to_file=filename)