import tensorflow as tf
from tensorflow.python.eager import context

from preprocessor.label_encoder import LabelEncoder

""" this is heavily inspired by
https://cs230-stanford.github.io/tensorflow-input-data.html
and
https://www.tensorflow.org/guide/datasets
"""


def _load_label(image_filename: str, label_filename: str):
    """Load the label from the filename(for both training and validation).

    The following operations are applied:
        - Load the label from the file
        - Split the label with , delimiter
        - reshape in 1 box = 1 row
    """
    label_string = tf.read_file(label_filename)
    label_split = tf.string_split([label_string], delimiter=',').values  # use .values to get tensor from sparse tensor
    label_int = tf.strings.to_number(label_split, out_type=tf.int32)
    label_reshaped = tf.reshape(label_int, shape=[-1, 5])
    return image_filename, label_reshaped


def _encode_label(image, label, encoder: LabelEncoder):
    """Encode

    """
    # return random data for the moment. will be fixed later
    scale1 = tf.random_uniform(shape=[38, 38, 4, 84], dtype=tf.float32)
    scale2 = tf.random_uniform(shape=[19, 19, 6, 84], dtype=tf.float32)
    scale3 = tf.random_uniform(shape=[10, 10, 6, 84], dtype=tf.float32)
    scale4 = tf.random_uniform(shape=[5, 5, 6, 84], dtype=tf.float32)
    scale5 = tf.random_uniform(shape=[3, 3, 4, 84], dtype=tf.float32)
    scale6 = tf.random_uniform(shape=[1, 1, 4, 84], dtype=tf.float32)
    encoded_label = (scale1, scale2, scale3, scale4, scale5, scale6)
    return image, encoded_label


def _load_image(filename: str, label, img_width: int, img_height: int):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(images=image,
                                           size=[img_width, img_height],
                                           preserve_aspect_ratio=False)
    return resized_image, label


def _augment_image(image, label, use_random_flip):
    """Image preprocessing for training.

    The following operations are applied:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_fn(is_training: bool, filenames: [], labels: [], params):
    """Input function for the SSD dataset.

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data/{label}_{id}.jpg"...]
        labels: (list) filenames of the labels, as ["data/{label}_{id}.txt"...]
        params: (Params) contains hyper-parameters of the model (ex: `params.num_epochs`)

    Hyper-parameter:
        num_classes: (int) asdf
        batch_size: (int) asdf
        image_width: (int) asdf
        image_height: (int)asdf
        num_parallel_calls: (int) asdf
        feature_map_sizes: (array) asdf
        ratios: (array) asdf
        iou: (float) asdf
        use_random_flip: (bool) asdfg
    """

    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # encoder = LabelEncoder(num_classes=params.num_classes,
    #                        img_width=params.image_width,
    #                        img_height=params.image_height,
    #                        feature_map_sizes=params.feature_map_sizes,
    #                        ratios=params.ratios,
    #                        iou_threshold=params.iou)
    encoder = None

    # Create a Dataset serving batches of images and labels
    load_label = lambda f, l: _load_label(f, l)
    encode_label = lambda f, l: _encode_label(f, l, encoder)
    load_img = lambda f, l: _load_image(f, l, params.image_width, params.image_height)
    augment_img = lambda f, l: _augment_image(f, l, params.use_random_flip)

    with tf.device('/cpu:0') and tf.variable_scope('feeding_data'):
        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                       .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
                       .repeat(params.num_epochs)  #
                       .map(load_label, num_parallel_calls=params.num_parallel_calls)
                       .map(encode_label, num_parallel_calls=params.num_parallel_calls)
                       .map(load_img, num_parallel_calls=params.num_parallel_calls)
                       .map(augment_img, num_parallel_calls=params.num_parallel_calls)
                       .batch(params.batch_size)
                       .prefetch(1)  # make sure you always have one batch ready to serve
                       )
        else:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                       .map(load_label)
                       .map(encode_label)
                       .map(load_img)
                       .batch(params.batch_size)
                       .prefetch(1)  # make sure you always have one batch ready to serve
                       )

        # if in eager mode, return iterator for testing
        if context.executing_eagerly():
            iterator = dataset.make_one_shot_iterator()
            return iterator

        # in regular mode create iterator init operation instead
        # Create re-initializable iterator from dataset
        iterator = dataset.make_initializable_iterator()
        iterator_init_op = iterator.initializer
        return iterator, iterator_init_op
