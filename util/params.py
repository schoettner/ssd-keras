import json

from dataclasses import dataclass

# https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
@dataclass
class Params:

    # input dimensions
    num_classes: int = 2
    img_width: int = 300
    img_height: int = 300

    # training parameters
    batch_size: int = 2
    num_epochs: int = 2
    num_parallel_calls: int = 4
    steps_per_epoch: int = 10
    validation_steps: int = 10
    learning_rate: float = 0.001
    use_eval: bool = False

    # augmentation params
    use_random_flip: bool = False

    # i/o params
    train_data_path: str = '/data/train'
    eval_data_path: str = './data/eval'
    log_dir: str = 'utput/log/'
    checkpoint_dir: str = 'output/checkpoints/'
    checkpoint_file: str = 'checkpoint.hdf5'
    weights_file: str = 'output/ssd-weights.h5'
    model_file: str = 'utput/ssd-model.hdf5'

    def __init__(self, file_path: str):
        self.__update_json__(file_path)

    def __update_json__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
