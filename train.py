import os
import sys

import tensorflow as tf

from model.loss import Loss
from model.ssd import SSD
from preprocessor.tf import tf_dataset
from preprocessor.np.pre_processor import PreProcessor
from preprocessor.tf.tf_dataset_spec import TfDatasetSpec
from util.params import Params
from util.utils import print_model


def main(params_path: str):
    tf.logging.set_verbosity(tf.logging.INFO)

    config = Params(file_path=params_path)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # init objects
    # generator = create_generator(config)
    iterator = create_iterator()

    ssd_model = SSD(config).build_model()
    print_model(ssd_model)
    ssd_loss = Loss()

    print('Compile Model for Training')
    ssd_model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate),
                      loss=ssd_loss.calculate_loss,
                      metrics=['acc'])
    ssd_model.summary()

    loss_to_monitor = 'val_loss' if config.use_eval else 'loss'
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=config.log_dir),
        tf.keras.callbacks.EarlyStopping(patience=2, monitor=loss_to_monitor),
        tf.keras.callbacks.ModelCheckpoint(filepath=config.checkpoint_dir + config.checkpoint_file,
                                           monitor=loss_to_monitor,
                                           verbose=0,
                                           save_best_only=True,
                                           mode='auto',
                                           period=1)
    ]

    print('Starting Training')
    # ssd_model.fit_generator(generator=generator,
    #                         steps_per_epoch=config['steps_per_epoch'],
    #                         callbacks=callbacks,
    #                         initial_epoch=0,
    #                         epochs=config['epochs'])

    # https://github.com/tensorflow/tensorflow/issues/20022
    ssd_model.fit(x=iterator,
                  steps_per_epoch=1,
                  callbacks=callbacks,
                  initial_epoch=0,
                  epochs=1,
                  )
    print('Training completed. Saving Model...')

    ssd_model.save_weights(config.weights_file, save_format='h5')
    ssd_model.save(config.model_file)
    print('Model saved. Training complete')


def create_generator(config):
    return PreProcessor(config).get_random_training_generator()


def create_iterator():
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    iterator, init_op = tf_dataset.input_fn(True,
                                            TfDatasetSpec.given_test_image(),
                                            TfDatasetSpec.given_test_labels(),
                                            TfDatasetSpec.given_test_params())
    sess.run(init_op)
    return iterator


if __name__ == "__main__":
    # could also use len(sys.argv) to check if there is an argument
    try:
        config_file = sys.argv[1]
    except IndexError:
        config_file = 'params.json'
    main(config_file)
