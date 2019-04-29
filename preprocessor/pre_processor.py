import numpy as np

from preprocessor.batch_loader import BatchLoader
from preprocessor.label_encoder import LabelEncoder


class PreProcessor(object):
    """
    provide a data generator to handle
    """
    def __init__(self, config: dict):
        self.config = config
        batch_size = self.config['batch_size']
        num_classes = self.config['num_classes']
        img_width = self.config['img_width']
        img_height = self.config['img_height']

        file_list = []
        self.batch_loader = BatchLoader(file_list, batch_size=batch_size)
        self.label_encoder = LabelEncoder(num_classes, img_width, img_height)

    def get_training_generator(self):
        batch = self.batch_loader.load_next_batch()

    def get_random_training_generator(self):
        batch_size = self.config['batch_size']
        img_width = self.config['img_width']
        img_height = self.config['img_height']

        while 1:

            x = np.random.rand(batch_size,
                               img_width,
                               img_height,
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
