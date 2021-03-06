import tensorflow.keras.backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Permute, Flatten, concatenate
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import Layer

from model.l2_normalization_layer import L2Normalization
from util.params import Params


class SSD:

    def __init__(self, params: Params, mode: str = 'train', base_network: str = 'vgg-16'):
        self.mode = mode
        self.base_network = base_network
        self.num_classes = params.num_classes
        self.img_width = params.img_width
        self.img_height = params.img_height
        self.channels = 3
        # default in ssd is: 1, 2, 3, 1/2, 1/3 == 6 boxes (incl s'k) and 6 scales
        self.aspect_ratios_per_layer = params.ratios
        self.num_bboxes_per_layer = []
        for layer in self.aspect_ratios_per_layer:
            if 1 in layer:
                # add s´k if 1 is included (ssd paper page 6)
                self.num_bboxes_per_layer.append(len(layer) + 1)
            else:
                self.num_bboxes_per_layer.append(len(layer))
        print('Boxes per scale: {}'.format(self.num_bboxes_per_layer))

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
            # 2.1 Model - multi-scale feature maps for detection (ssd paper page 3)
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

            conv4_3_norm = L2Normalization(name='conv4_3_norm')(conv4_3)

            ### instead of creating blocks with a permute and flatten, we only use the plain conv  at this point

            # create the location for each feature map. each box has 4 parameters ∆(cx, cy, w, h)
            # Output shape of the localization layers: (batch, feature_map_width, feature_map_height, n_boxes * 4)
            conv4_3_norm_mbox_loc = Conv2D(self.num_bboxes_per_layer[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
            fc7_mbox_loc = Conv2D(self.num_bboxes_per_layer[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
            conv6_2_mbox_loc = Conv2D(self.num_bboxes_per_layer[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
            conv7_2_mbox_loc = Conv2D(self.num_bboxes_per_layer[3] * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
            conv8_2_mbox_loc = Conv2D(self.num_bboxes_per_layer[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
            conv9_2_mbox_loc = Conv2D(self.num_bboxes_per_layer[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)

            # create the confidence for each feature map. each box has the same number of classes (c1, c2, · · · , cp).
            # use softmax activation
            # Output shape of the confidence layers: (batch, feature_map_width, feature_map_height, self.num_bboxes_per_layer * num_classes)
            conv4_3_norm_mbox_conf = Conv2D(self.num_bboxes_per_layer[0] * self.num_classes, (3, 3), activation="softmax", padding='same', name='conv4_3_norm_mbox_conf')(conv4_3_norm)
            fc7_mbox_conf = Conv2D(self.num_bboxes_per_layer[1] * self.num_classes, (3, 3), activation="softmax", padding='same', name='fc7_mbox_conf')(fc7)
            conv6_2_mbox_conf = Conv2D(self.num_bboxes_per_layer[2] * self.num_classes, (3, 3), activation="softmax", padding='same', name='conv6_2_mbox_conf')(conv6_2)
            conv7_2_mbox_conf = Conv2D(self.num_bboxes_per_layer[3] * self.num_classes, (3, 3), activation="softmax", padding='same', name='conv7_2_mbox_conf')(conv7_2)
            conv8_2_mbox_conf = Conv2D(self.num_bboxes_per_layer[4] * self.num_classes, (3, 3), activation="softmax", padding='same', name='conv8_2_mbox_conf')(conv8_2)
            conv9_2_mbox_conf = Conv2D(self.num_bboxes_per_layer[5] * self.num_classes, (3, 3), activation="softmax", padding='same', name='conv9_2_mbox_conf')(conv9_2)

            # reshape the conv to bounding box index
            # the the first two dimensions from their input shape, third dimension = number of boxes per layer, last dimension = amount of values (4 or num classes)
            scale1_loc = Reshape([*K.int_shape(conv4_3_norm_mbox_loc)[1:3], self.num_bboxes_per_layer[0], 4], name='reshape_scale1_loc')(conv4_3_norm_mbox_loc)
            scale2_loc = Reshape([*K.int_shape(fc7_mbox_loc)[1:3], self.num_bboxes_per_layer[1], 4], name='reshape_scale2_loc')(fc7_mbox_loc)
            scale3_loc = Reshape([*K.int_shape(conv6_2_mbox_loc)[1:3], self.num_bboxes_per_layer[2], 4], name='reshape_scale3_loc')(conv6_2_mbox_loc)
            scale4_loc = Reshape([*K.int_shape(conv7_2_mbox_loc)[1:3], self.num_bboxes_per_layer[3], 4], name='reshape_scale4_loc')(conv7_2_mbox_loc)
            scale5_loc = Reshape([*K.int_shape(conv8_2_mbox_loc)[1:3], self.num_bboxes_per_layer[4], 4], name='reshape_scale5_loc')(conv8_2_mbox_loc)
            scale6_loc = Reshape([*K.int_shape(conv9_2_mbox_loc)[1:3], self.num_bboxes_per_layer[5], 4], name='reshape_scale6_loc')(conv9_2_mbox_loc)
            scale1_conf = Reshape([*K.int_shape(conv4_3_norm_mbox_conf)[1:3], self.num_bboxes_per_layer[0], self.num_classes], name='reshape_scale1_conf')(conv4_3_norm_mbox_conf)
            scale2_conf = Reshape([*K.int_shape(fc7_mbox_conf)[1:3], self.num_bboxes_per_layer[1], self.num_classes], name='reshape_scale2_conf')(fc7_mbox_conf)
            scale3_conf = Reshape([*K.int_shape(conv6_2_mbox_conf)[1:3], self.num_bboxes_per_layer[2], self.num_classes], name='reshape_scale3_conf')(conv6_2_mbox_conf)
            scale4_conf = Reshape([*K.int_shape(conv7_2_mbox_conf)[1:3], self.num_bboxes_per_layer[3], self.num_classes], name='reshape_scale4_conf')(conv7_2_mbox_conf)
            scale5_conf = Reshape([*K.int_shape(conv8_2_mbox_conf)[1:3], self.num_bboxes_per_layer[4], self.num_classes], name='reshape_scale5_conf')(conv8_2_mbox_conf)
            scale6_conf = Reshape([*K.int_shape(conv9_2_mbox_conf)[1:3], self.num_bboxes_per_layer[5], self.num_classes], name='reshape_scale6_conf')(conv9_2_mbox_conf)

            # do not use the prior boxes here. instead, create multiple scales. each scale has an own output
            # hint: do not use the Concatenate layer since it seems to have an bug. use the concatenate function instead
            # this output is motivated on the regular YOLOv3 network
            scale1 = concatenate([scale1_loc, scale1_conf], axis=4, name='scale1_prediction')
            scale2 = concatenate([scale2_loc, scale2_conf], axis=4, name='scale2_prediction')
            scale3 = concatenate([scale3_loc, scale3_conf], axis=4, name='scale3_prediction')
            scale4 = concatenate([scale4_loc, scale4_conf], axis=4, name='scale4_prediction')
            scale5 = concatenate([scale5_loc, scale5_conf], axis=4, name='scale5_prediction')
            scale6 = concatenate([scale6_loc, scale6_conf], axis=4, name='scale6_prediction')
            return [scale1, scale2, scale3, scale4, scale5, scale6]

        else:
            raise NotImplementedError('The selected base network: %s, is not supported ' % self.base_network)

    @staticmethod
    def build_legacy_mbox_block(block_input: Layer,
                                num_output: int,
                                block_name: str,
                                block_type: str = 'loc'):
        """
        builds the mbox block as described in the caffe implementation in train.prototext (starting in line 955)
        however, the permute and flatten are not used in this implementation.

        :param block_input: input layer of the block
        :param num_output: amount of filters in the convolution layer = number of expected outputs
        :param block_name: the name of the block. given by the input convolution = the feature map of the block
        :param block_type: 'loc' for the location block, 'conf' for the confidence block
        :return: the output layer of the block as flat vector
        """
        mbox = Conv2D(num_output, (3, 3), activation=None, padding='same', name='{}_mbox_{}'.format(block_name, block_type))(block_input)
        mbox_perm = Permute(dims=(1, 3, 2), name='{}_mbox_{}_perm'.format(block_name, block_type))(mbox)
        mbox_flat = Flatten(name='{}_mbox_{}_flat'.format(block_name, block_type))(mbox_perm)
        return mbox_flat
