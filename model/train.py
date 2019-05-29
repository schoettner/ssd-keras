import logging
import os
import tensorflow as tf

def train_and_evaluate(train_model,
                       eval_model,
                       params,
                       output_dir: str = './output'):
    last_saver = tf.train.Saver()  # save last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # save best checkpoint
    begin_at_epoch = 0

    with tf.Session() as sess:
        # init model variables
        sess.run(train_model['variable_init_op'])

        # setup tensorboard
        train_writer = tf.summary.FileWriter(os.path.join(output_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(output_dir, 'eval_summaries'), sess.graph)

        best_eval_accuracy = 0.0

        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epchs):
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))
            num_training_steps = (params.train_size + params.batch_size - 1) // params.batch_size

            # Save weights
            last_save_path = os.path.join(output_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

def train_sess():
    return None