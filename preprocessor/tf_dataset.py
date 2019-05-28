import tensorflow as tf

from preprocessor.label_encoder import LabelEncoder


def _load_label(image_filename: str, label_filename: str):
    """Load the label from the filename(for both training and validation).

    The following operations are applied:
        - Load the label from the file
        - Convert to the label from class_1, x_min_1, y_min_1, x_max_1, y_max_1, class_2, ...
          to class_1, x_center_1, y_center_1, w_1, h_1, class_2
    """
    label = None
    return image_filename, label


def _encode_label(image, label, encoder: LabelEncoder):
    encoded_label = encoder.convert_label(label)
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


def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    encoder = LabelEncoder(num_classes=params.num_classes,
                           img_width=params.image_width,
                           img_height=params.image_height,
                           feature_map_sizes=params.feature_map_sizes,
                           ratios=params.ratios,
                           iou_threshold=params.iou)

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    load_label = lambda f, l: _load_label(f, l)
    encode_label = lambda f, l: _encode_label(f, l, encoder)
    load_img = lambda f, l: _load_image(f, l, params.image_width, params.image_height)
    augment_img = lambda f, l: _augment_image(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
                   .map(load_label, num_parallel_calls=params.num_parallel_calls)
                   .map(encode_label, num_parallel_calls=params.num_parallel_calls)
                   .map(load_img, num_parallel_calls=params.num_parallel_calls)
                   .map(augment_img, num_parallel_calls=params.num_parallel_calls)
                   .batch(params.batch_size)
                   .prefetch(3)  # make sure you always have three batches ready to serve
                   )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(load_img)
                   .batch(params.batch_size)
                   .prefetch(1)  # make sure you always have one batch ready to serve
                   )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
