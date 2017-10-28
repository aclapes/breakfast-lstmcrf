import numpy as np
import tensorflow as tf
from progressbar import ProgressBar

import src.crf as crf  # master's version of tf.contrib.crf
from src.evaluation import compute_framewise_accuracy
from src.reader import read_data_generator


class CrfPipeline(object):
    def __init__(self,
                 train,
                 val,
                 te,
                 no_classes,
                 batch_size,
                 learn_rate,
                 num_epochs,
                 optimizer_type='adam'):

        self.x, self.y, self.lengths = train['video_features'], train['outputs'], train['lengths']
        self.x_val, self.y_val, self.lengths_val = val['video_features'], val['outputs'], val['lengths']
        self.x_te, self.y_te, self.lengths_te = te['video_features'], te['outputs'], te['lengths']

        self.no_classes = no_classes

        self.batch_size = batch_size
        self.num_words = train['video_features'].shape[1]
        self.num_features = train['video_features'].shape[2]

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs

        self.optimizer_type = optimizer_type

        self.graph = tf.Graph()
        with self.graph.as_default():
            # x = features, y = labels in one-hot encoding, and w = binary mask of valid timesteps
            self.x_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_words, self.num_features])
            self.y_batch = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_words])
            self.w_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_words])
            # get sequences length from binary mask of valid timesteps
            self.lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

            self.x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
            x_drop = tf.nn.dropout(self.x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

            # Compute unary scores from a linear layer.
            matricied_x = tf.reshape(x_drop, [-1, self.num_features])
            softmax_w = tf.get_variable('softmax_w', [self.num_features, self.no_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.no_classes], dtype=tf.float32)
            logits = tf.matmul(matricied_x, softmax_w) + softmax_b

            normalized_logits = tf.nn.softmax(logits)
            self.unary_scores = tf.reshape(
                normalized_logits, [self.batch_size, self.num_words, self.no_classes]
            )

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time.
            self.log_likelihood, self.transition_params = crf.crf_log_likelihood(
                self.unary_scores, self.y_batch, self.lengths_batch)
            # Add a training op to tune the parameters.

            self.decoding, _ = crf.crf_decode(self.unary_scores, self.transition_params, self.lengths_batch)

            self.loss = tf.reduce_mean(-self.log_likelihood)

            # self.apply_placeholder_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

            if self.optimizer_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=0.01, scope=None
            )
            weights = tf.trainable_variables()
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            # self.grads = tf.gradients(self.loss, tf.trainable_variables())
            # self.clip_grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
            # self.apply_placeholder_op = self.optimizer.apply_gradients(zip(self.clip_grads, tf.trainable_variables()))
            regularized_loss = self.loss + regularization_penalty
            self.apply_placeholder_op = self.optimizer.minimize(regularized_loss)

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()  # always session.run this op first!

    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e + 1, self.num_epochs))
                self.train_reader = read_data_generator(
                    self.x, self.y, self.lengths, batch_size=self.batch_size
                )

                num_train_batches = self.x.shape[0] // self.batch_size
                train_batch_loss = [None] * num_train_batches
                train_batch_accs = [None] * num_train_batches

                progbar = ProgressBar(max_value=self.x.shape[0] // self.batch_size)
                for b in range(num_train_batches):
                    batch = self.train_reader.next()

                    # Run forward and backward (backprop)
                    train_batch_loss[b], decoding, _ = session.run(
                        [self.loss, self.decoding, self.apply_placeholder_op],
                        feed_dict={
                            self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2]
                        }
                    )
                    train_batch_accs[b] = compute_framewise_accuracy(decoding, batch[1], batch[2])

                    progbar.update(b)
                progbar.finish()

                # Validation
                self.val_reader = read_data_generator(
                    self.x_val, self.y_val, self.lengths_val, batch_size=self.batch_size
                )

                num_val_batches = self.x_val.shape[0] // self.batch_size
                val_batch_loss = [None] * num_val_batches
                val_batch_accs = [None] * num_val_batches

                progbar = ProgressBar(max_value=self.x_val.shape[0] // self.batch_size)
                for b in range(num_val_batches):
                    batch = self.val_reader.next()

                    # Run forward, but not backprop
                    val_batch_loss[b], decoding = session.run(
                        [self.loss, self.decoding],
                        feed_dict={self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2]}
                    )
                    val_batch_accs[b] = compute_framewise_accuracy(decoding, batch[1], batch[2])

                    progbar.update(b)
                progbar.finish()

                # Validation accuracy is the mean accuracy over batch accuracies
                print(
                    'TRAIN (loss/acc): %.4f/%.2f%%, VAL (loss/acc): %.4f/%.2f%%' % (
                        np.mean(train_batch_loss), np.mean(train_batch_accs),
                        np.mean(val_batch_loss), np.mean(val_batch_accs)
                    )
                )

                if e % 2 == 0:
                    self.saver.save(session, 'simplecrf_model', global_step=e)

            # Testing
            self.te_reader = read_data_generator(self.x_te, self.y_te, self.lengths_te, batch_size=1)

            init_state = np.zeros((2, 1, self.hidden_size), dtype=np.float32)  # 2 for c and h

            num_te_batches = self.x_te.shape[0]  # batch_size = 1 in this case
            te_batch_accs = [None] * num_te_batches

            progbar = ProgressBar(max_value=num_te_batches)
            for b in range(num_te_batches):
                batch = self.te_reader.next()

                # Run forward, but not backprop
                decoding = session.run(
                    [self.pred],
                    feed_dict={
                        self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2],
                        self.state_placeholder: init_state
                    }
                )
                te_batch_accs[b] = compute_framewise_accuracy(decoding, batch[1], batch[2])
                progbar.update(b)
            progbar.finish()

            print(
                'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' %
                (np.mean(train_batch_accs), np.mean(val_batch_accs), np.mean(te_batch_accs))
            )
