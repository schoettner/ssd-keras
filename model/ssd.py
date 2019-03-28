from tensorflow.python.keras import Input
from tensorflow.keras.applications import ResNet50


class SSD:

    def __init__(self, mode: str = 'train'):
        self.mode = mode
        self.num_classes = 80
        self.img_width = 224
        self.img_height = 224
        self.channels = 3

    def build_model(self):
        Input(shape=(self.img_width, self.img_height, self.channels))
        model = ResNet50(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         classes=self.num_classes)
        return model
