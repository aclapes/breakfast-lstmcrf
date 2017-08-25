import numpy as np
import tensorflow as tf

from progressbar import ProgressBar

import src.crf as crf  # master's version of tf.contrib.crf

from reader import read_data_generator
from evaluation import compute_framewise_accuracy


class SimpleCrfModel(object):
    def __init__(self, config, input_data, is_training):
        self.config = config
        self.input_data = input_data
        self.is_training = is_training

        no_classes = config['no_classes']
        batch_size = config['batch_size']
        num_words = config['num_words']
        num_features = config['num_features']
        optimizer_type = config['optimizer_type']
        learn_rate = config['learn_rate']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[batch_size, num_words, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[batch_size, num_words])
        self.w_batch = tf.placeholder(tf.float32, shape=[batch_size, num_words])
        # get sequences length from binary mask of valid timesteps
        lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

        x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        if is_training:
            x_batch = tf.nn.dropout(x_batch, keep_prob=0.8)  # TODO: experiment with this dropout

        # Compute unary scores from a linear layer.
        matricied_x = tf.reshape(x_batch, [-1, num_features])
        softmax_w = tf.get_variable('softmax_w', [num_features, no_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b

        normalized_logits = tf.nn.softmax(logits)
        unary_scores = tf.reshape(
            normalized_logits, [batch_size, num_words, no_classes]
        )

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = crf.crf_log_likelihood(
            unary_scores, self.y_batch, lengths_batch)
        # Add a training op to tune the parameters.

        self.decoding, _ = crf.crf_decode(unary_scores, transition_params, lengths_batch)
        self.loss = tf.reduce_mean(-log_likelihood)

        if not is_training:
            return

        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

    def run_epoch(self, session):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''
        x = self.input_data['video_features']
        y = self.input_data['outputs']
        l = self.input_data['lengths']

        self.reader = read_data_generator(x, y, l, batch_size=self.config['batch_size'])

        num_batches = x.shape[0] // self.config['batch_size']
        batch_loss = [None] * num_batches
        batch_accs = [None] * num_batches

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = self.reader.next()

            fetches = {
                'loss' : self.loss,
                'decoding' : self.decoding
            }
            if self.is_training:
                fetches['train_op'] = self.train_op

            vals = session.run(
                fetches,
                feed_dict={self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2]
                }
            )
            batch_loss[b] = vals['loss']
            batch_accs[b] = compute_framewise_accuracy(vals['decoding'], batch[1], batch[2])

            progbar.update(b)
        progbar.finish()

        return np.mean(batch_loss), np.mean(batch_accs)


class SimpleCrfPipeline(object):
    def __init__(self,
                 train,
                 val,
                 te,
                 no_classes,
                 batch_size,
                 learn_rate,
                 num_epochs,
                 optimizer_type='adam'):

        self.num_epochs = num_epochs

        config = dict(
            no_classes = no_classes,
            batch_size = batch_size,
            num_words = train['video_features'].shape[1],
            num_features = train['video_features'].shape[2],
            optimizer_type = optimizer_type,
            learn_rate = learn_rate
        )

        test_config = config.copy()
        test_config['batch_size'] = 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-0.05, 0.05)
            with tf.variable_scope('Model', reuse=False, initializer=initializer):
                self.train_model = SimpleCrfModel(config=config, input_data=train, is_training=True)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.val_model = SimpleCrfModel(config=config, input_data=val, is_training=False)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.te_model = SimpleCrfModel(config=test_config, input_data=te, is_training=False)

            self.init_op = tf.global_variables_initializer()


    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e + 1, self.num_epochs))
                train_eval = self.train_model.run_epoch(session)
                val_eval = self.val_model.run_epoch(session)
                print(
                    'TRAIN (loss/acc): %.4f/%.2f%%, VAL (loss/acc): %.4f/%.2f%%' % (
                        train_eval[0], train_eval[1], val_eval[0], val_eval[1]
                    )
                )
            te_eval = self.te_model.run_epoch(session)

            print(
                'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' % (
                    train_eval[1], val_eval[1], te_eval[1])
            )