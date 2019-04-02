import os
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from model.loss import Loss
from model.ssd import SSD


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    config = load_config_file()
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    generator = create_generator(config)
    ssd_model = SSD().build_model()
    print_model(ssd_model)
    ssd_loss = Loss()

    print('Compile Model for Training')
    ssd_model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                      loss=ssd_loss.calculate_loss,
                      metrics=['acc'])
    ssd_model.summary()

    loss_to_monitor = 'val_los' if config['use_eval'] else 'loss'
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=config['log_dir']),
        tf.keras.callbacks.EarlyStopping(patience=2,
                                         monitor=loss_to_monitor),
        tf.keras.callbacks.ModelCheckpoint(filepath=config['checkpoint_dir']+config['checkpoint_file'],
                                           monitor=loss_to_monitor,
                                           verbose=0,
                                           save_best_only=True,
                                           mode='auto',
                                           period=1)
    ]

    print('Starting Training')
    ssd_model.fit_generator(generator=generator,
                            steps_per_epoch=config['steps_per_epoch'],
                            callbacks=callbacks,
                            initial_epoch=0,
                            epochs=config['epochs'])
    print('Training completed. Saving Model...')

    ssd_model.save_weights(config['weights_file'], save_format='h5')
    ssd_model.save(config['model_file'])
    print('Model saved. Training complete')


def create_generator(config):
    batch_size = config['batch_size']
    while 1:

        x = np.random.rand(batch_size,
                           config['img_width'],
                           config['img_height'],
                           3)

        # y = np.random.randint(config['num_classes'],
        #                       size=config['batch_size'])
        y1 = np.random.rand(batch_size, 38, 38, 4, 84)
        y2 = np.random.rand(batch_size, 19, 19, 6, 84)
        y3 = np.random.rand(batch_size, 10, 10, 6, 84)
        y4 = np.random.rand(batch_size, 5, 5, 6, 84)
        y5 = np.random.rand(batch_size, 3, 3, 4, 84)
        y6 = np.random.rand(batch_size, 1, 1, 4, 84)
        yield x, [y1, y2, y3, y4, y5, y6]


def load_config_file(config_file: str = 'config.yml'):
    with open(config_file, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as ex:
            print(ex)


def print_model(model, filename='model.png'):
    plot_model(model, to_file=filename)


if __name__ == "__main__":
    main()
