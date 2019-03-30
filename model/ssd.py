from tensorflow.python.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Permute, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import Layer

from model.l2_normalization import L2Normalization


class SSD:

    def __init__(self, mode: str = 'train', base_network: str = 'vgg-16'):
        self.mode = mode
        self.base_network = base_network
        self.num_classes = 80
        self.img_width = 300
        self.img_height = 300
        self.channels = 3

    def get_resnet(self):
        """
        return a resnet 50 for demo purposes
        """
        model = ResNet50(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         classes=self.num_classes)
        return model

    def build_model(self):
        model_input = Input(shape=(self.img_width, self.img_height, self.channels))
        base_network = self.build_base_network(input=model_input)
        return Model(inputs=model_input, outputs=base_network)

    def build_base_network(self, input: Input):
        if self.base_network == 'vgg-16':

            # the architecture is taken from the train.prototxt from the origin caffe implementation
            # there are some modifications to the regular VGG16 (SSD Base network section on page 7)
            conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same',  name='conv1_1')(input)
            conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same',  name='conv1_2')(conv1_1)
            pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

            conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
            conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
            pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

            conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
            conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
            conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
            pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

            conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
            conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
            conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
            pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

            conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
            conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
            conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
            pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

            fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
            fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same',  name='fc7')(fc6)
            # fc7 is the last layer from the base network.

            # further conv layer for more scales
            conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
            conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
            conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv6_2')(conv6_1)

            conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
            conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
            conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_1)

            conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
            conv8_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)

            conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
            conv9_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)

            # train.prototxt line 940
            conv4_3_norm = L2Normalization(name='conv4_3_norm')(conv4_3)

            # mbox_loc
            conv4_3_norm_loc_block = self.build_mbox_block(block_input=conv4_3_norm,
                                                           num_output=16,
                                                           block_name='conv4_3_norm',
                                                           block_type='loc')

            # mbox_conf
            conv4_3_conf_block = self.build_mbox_block(block_input=conv4_3_norm,
                                                       num_output=84,
                                                       block_name='conv4_3_norm',
                                                       block_type='conf')

            return [conv9_2, conv4_3_norm_loc_block, conv4_3_conf_block]

        else:
            raise NotImplementedError('The selected base network: %s, is not supported ' % self.base_network)

    def build_mbox_block(self,
                         block_input: Layer,
                         num_output: int,
                         block_name: str,
                         block_type: str = 'loc',
                         ):
        mbox = Conv2D(num_output, (3, 3), activation=None, padding='same', name='{}_mbox_{}'.format(block_name, block_type))(block_input)
        mbox_perm = Permute(dims=(1, 3, 2), name='{}_mbox_{}_perm'.format(block_name, block_type))(mbox)
        mbox_flat = Flatten(name='{}_mbox_{}_flat'.format(block_name, block_type))(mbox_perm)
        return mbox_flat
