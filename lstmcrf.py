import numpy as np
import tensorflow as tf

import h5py
import argparse
from progressbar import ProgressBar

from tensorflow.contrib import rnn
import src.crf as crf  # master's version of tf.contrib.crf


def read_data_generator(data, labels, lengths, batch_size=16):
    '''
    This generator function serves a batch of the dataset at each call.
    See what a generator function is ;)
    :param data:
    :param labels:
    :param lengths:
    :param batch_size:
    :return:
    '''

    n_batches = len(data) // batch_size  # this will discard the last batch

    for i in range(n_batches):
        # prepare the batch
        x = data[(i*batch_size):((i+1)*batch_size),:,:] # batch features
        y = labels[(i * batch_size):((i + 1) * batch_size), :] # batch labels

        # instead of using lengths, create a binary mask (to mask padded timesteps)
        l = lengths[(i * batch_size):((i + 1) * batch_size)]  # not returned!

        # batch mask preparation
        w = np.zeros((l.shape[0], y.shape[1]), dtype=np.float32)
        for k in range(l.shape[0]):
            l_k = int(l[k])  # length of the k-th seq in the batch
            w[k, :l_k] = 1.  # binary mask for k-th seq

        yield (x, y, w)


def compute_framewise_accuracy(predictions, labels, weights):
    '''
    Computes the framewise accuracy over a set of predictions.

    :param predictions: 2-D array of predictions [num_batches, num_timesteps]
    :param labels: 2-D array of labels [num_batches, num_timesteps]
    :param weights: 2-D array of weights [num_batches, num_timesteps]. Used to mask padded timesteps.
    :return:
    '''

    correct_labels = total_labels = 0.

    for pred, y, w in zip(predictions, labels, weights):
        length = int(np.sum(w))
        correct_labels += np.sum(np.equal(pred[:length], y[:length]))
        total_labels += length

    return 100. * correct_labels / float(total_labels)


class SimpleCrfModel(object):
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


class SimpleLstmCrfModel(object):
    def __init__(self,
                 train,
                 val,
                 te,
                 no_classes,
                 batch_size,
                 learn_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam'):

        self.x, self.y, self.lengths = train['video_features'], train['outputs'], train['lengths']
        self.x_val, self.y_val, self.lengths_val = val['video_features'], val['outputs'], val['lengths']

        self.no_classes = no_classes

        self.batch_size = batch_size
        self.num_words = self.x.shape[1]
        self.num_features = self.x.shape[2]

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size

        self.optimizer_type = optimizer_type

        self.graph = tf.Graph()
        with self.graph.as_default():
            # x = features, y = labels in one-hot encoding, and w = binary mask of valid timesteps
            self.x_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_words, self.num_features])
            self.y_batch = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_words])
            self.w_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_words])
            # get sequences length from binary mask of valid timesteps
            self.lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

            # lstm state from previous iterations (stateful lstm)
            self.state_placeholder = tf.placeholder(tf.float32, [2, self.batch_size, self.hidden_size])

            self.x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
            x_drop = tf.nn.dropout(self.x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

            # set the statefulness (if it weren't check other instruction)
            # </--
            self.init_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])
            # self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            # ---/>

            cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-self.drop_prob)  # TODO: check this
            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(
                cell,
                x_drop,
                dtype=tf.float32,
                initial_state=self.init_state_tuple, # statefull rnn
                sequence_length=self.lengths_batch  # do not process padded parts

            )

            # outputs_dropout = tf.nn.dropout(rnn_outputs, keep_prob=(1-self.drop_prob))

            # Compute unary scores from a linear layer.
            output = tf.reshape(rnn_outputs, [-1, self.hidden_size])
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, self.no_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.no_classes], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b

            normalized_logits = tf.nn.softmax(logits)
            self.unary_scores = tf.reshape(normalized_logits, [self.batch_size, self.num_words, self.no_classes])

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time.
            self.log_likelihood, self.transition_params = crf.crf_log_likelihood(
                self.unary_scores, self.y_batch, self.lengths_batch)
            # Add a training op to tune the parameters.

            self.decoding, _ = crf.crf_decode(self.unary_scores, self.transition_params, self.lengths_batch)

            self.loss = tf.reduce_mean(-self.log_likelihood)

            # self.apply_placeholder_op = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

            if self.optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

            # l1_regularizer = tf.contrib.layers.l1_regularizer(
            #     scale=0.01, scope=None
            # )
            # weights = tf.trainable_variables()
            # regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            tvars = tf.trainable_variables()
            # self.grads = tf.gradients(self.loss, tf.trainable_variables())
            # self.clip_grads = [tf.clip_by_value(g, -1, 1) for g in self.grads]
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
            self.apply_placeholder_op = optimizer.apply_gradients(zip(grads, tvars))
            # regularized_loss = self.loss + regularization_penalty

            # self.apply_placeholder_op = self.optimizer.minimize(regularized_loss)

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()  # always session.run this op first!

    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e+1,self.num_epochs))

                self.train_reader = read_data_generator(
                    self.x, self.y, self.lengths, batch_size=self.batch_size
                )

                num_train_batches = self.x.shape[0] // self.batch_size
                train_batch_loss = [None] * num_train_batches
                train_batch_accs = [None] * num_train_batches

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                progbar = ProgressBar(max_value=num_train_batches)
                for b in range(num_train_batches):
                    batch = self.train_reader.next()

                    # Run forward and backward (backprop)
                    train_batch_loss[b], decoding, _, stateful_state = session.run(
                        [self.loss,
                         self.decoding,
                         self.apply_placeholder_op,
                         self.rnn_state],
                        feed_dict = {
                            self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                            self.state_placeholder : stateful_state
                        }
                    )
                    train_batch_accs[b] = compute_framewise_accuracy(decoding, batch[1], batch[2])
                    progbar.update(b)
                progbar.finish()

                # Validation
                self.val_reader = read_data_generator(
                    self.x_val, self.y_val, self.lengths_val, batch_size=self.batch_size
                )

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                num_val_batches = self.x_val.shape[0] // self.batch_size
                val_batch_accs = [None] * num_val_batches
                val_batch_loss = [None] * num_val_batches

                progbar = ProgressBar(max_value=num_val_batches)
                for b in range(num_val_batches):
                    batch = self.val_reader.next()

                    val_batch_loss[b], decoding, stateful_state = session.run(
                        [self.loss, self.decoding, self.rnn_state],
                        feed_dict={self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                                   self.state_placeholder : stateful_state}
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
                    self.saver.save(session, 'simplelstmcrf_model', global_step=e)

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
                    [self.decoding],
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


class SimpleLstmModel(object):
    def __init__(self,
                 train,
                 val,
                 test,
                 no_classes,
                 batch_size,
                 learn_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam'):

        self.x, self.y, self.lengths = train['video_features'], train['outputs'], train['lengths']
        self.x_val, self.y_val, self.lengths_val = val['video_features'], val['outputs'], val['lengths']
        self.x_te, self.y_te, self.lengths_te = test['video_features'], test['outputs'], test['lengths']

        self.no_classes = no_classes

        self.batch_size = batch_size
        self.num_words = self.x.shape[1]
        self.num_features = self.x.shape[2]

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size

        self.optimizer_type = optimizer_type

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_batch = tf.placeholder(tf.float32, shape=[None, self.num_words, self.num_features])
            self.y_batch = tf.placeholder(tf.int32, shape=[None, self.num_words])
            self.w_batch = tf.placeholder(tf.float32, shape=[None, self.num_words])

            self.lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

            self.state_placeholder = tf.placeholder(tf.float32, [2, None, self.hidden_size])

            # self.lrate = tf.placeholder(tf.float32, shape=[])

            self.x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
            x_drop = tf.nn.dropout(self.x_batch, keep_prob=1.0) # TODO: experiemnt with this one

            cell = rnn.LSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.drop_prob))

            init_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])
            # self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Obtain the idx-th from num_steps chunks of length step_size
            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(
                cell,
                x_drop,
                dtype=tf.float32,
                initial_state=init_state_tuple, # statefull rnn
                sequence_length=self.lengths_batch  # do not process padded parts
            )

            outputs_dropout = tf.nn.dropout(rnn_outputs, keep_prob=(1-self.drop_prob))

            # # Compute unary scores from a linear layer.
            # output = tf.reshape(tf.stack(self._outputs, axis=1), [-1, self.hidden_size])
            output = tf.reshape(outputs_dropout, [-1, self.hidden_size])
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, self.no_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
            logits = tf.matmul(output, softmax_w) + softmax_b

            normalized_logits = tf.nn.softmax(logits)
            normalized_logits = tf.reshape(
                normalized_logits, [tf.shape(self.x_batch)[0], self.num_words, self.no_classes]
            )

            self.pred = tf.argmax(normalized_logits, 2)

            loss = tf.contrib.seq2seq.sequence_loss(
                normalized_logits,
                self.y_batch,
                self.w_batch,
                average_across_timesteps=False,
                average_across_batch=True
            )
            self.cost = tf.reduce_sum(loss)

            if self.optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5.0)
            self.apply_placeholder_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()  # always session.run this op first!

    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e+1,self.num_epochs))

                self.train_reader = read_data_generator(
                    self.x, self.y, self.lengths, batch_size=self.batch_size
                )

                num_train_batches = self.x.shape[0] // self.batch_size
                train_batch_costs = [None] * num_train_batches
                train_batch_accs = [None] * num_train_batches

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                progbar = ProgressBar(max_value=num_train_batches)

                for b in range(num_train_batches):
                    batch = self.train_reader.next()

                    # Run forward and backward (backprop)
                    train_batch_costs[b], predictions, _, stateful_state = session.run(
                        [self.cost, self.pred, self.apply_placeholder_op, self.rnn_state],
                        feed_dict = {
                            self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                            self.state_placeholder : stateful_state
                        }
                    )
                    train_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                    progbar.update(b)
                progbar.finish()

                # Validation
                self.val_reader = read_data_generator(self.x_val, self.y_val, self.lengths_val, batch_size=self.batch_size)

                num_val_batches = self.x_val.shape[0] // self.batch_size
                val_batch_costs = [None] * num_val_batches
                val_batch_accs = [None] * num_val_batches

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                progbar = ProgressBar(max_value=num_val_batches)
                for b in range(num_val_batches):
                    batch = self.val_reader.next()

                    # Run forward, but not backprop
                    val_batch_costs[b], predictions, stateful_state = session.run(
                        [self.cost, self.pred, self.rnn_state],
                        feed_dict = {
                            self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                            self.state_placeholder : stateful_state
                        }
                    )
                    val_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                    progbar.update(b)
                progbar.finish()

                # Validation accuracy is the mean accuracy over batch accuracies
                print(
                    'TRAIN (cost/acc): %.4f/%.2f%%, VAL (cost/acc): %.4f/%.2f%%' % (
                        np.mean(train_batch_costs), np.mean(train_batch_accs),
                        np.mean(val_batch_costs), np.mean(val_batch_accs)
                    )
                )

                if e % 2 == 0:
                    self.saver.save(session, 'simplelstm_model', global_step=e)

            # Testing
            self.te_reader = read_data_generator(self.x_te, self.y_te, self.lengths_te, batch_size=1)

            init_state = np.zeros((2, 1, self.hidden_size), dtype=np.float32)  # 2 for c and h

            num_te_batches = self.x_te.shape[0]
            te_batch_accs = [None] * num_te_batches

            progbar = ProgressBar(max_value=num_te_batches)
            for b in range(num_te_batches):
                batch = self.te_reader.next()

                # Run forward, but not backprop
                loss, predictions = session.run(
                    [self.loss, self.pred],
                    feed_dict={
                        self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2],
                        self.state_placeholder: init_state
                    }
                )
                te_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                progbar.update(b)
            progbar.finish()

            print(
                'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' %
                (np.mean(train_batch_accs), np.mean(val_batch_accs), np.mean(te_batch_accs))
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform labelling of sequences using a LSTMCRF model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        dest='input_file',
        default='/data/datasets/breakfast/fv/s1/dataset.8-20.h5',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=32,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=1e-4,
        help=
        'Learning rate (default: %(default)s)')

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=2000,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-ot',
        '--optimizer-type',
        type=str,
        dest='optimizer_type',
        default='adam',
        help=
        'Optimizer type (sgd or adam) (default: %(default)s)')

    parser.add_argument(
        '-M',
        '--model-type',
        type=str,
        dest='model_type',
        default='lstmcrf',
        help=
        'Model type (crf, lstm or lstmcrf) (default: %(default)s)')

    # -----------------------------------------------
    # (LSTM-only parameters)
    # -----------------------------------------------
    parser.add_argument(
        '-s',
        '--hidden-size',
        type=int,
        dest='hidden_size',
        default=512,
        help=
        'Hidden size (default: %(default)s)')

    parser.add_argument(
        '-p',
        '--drop-prob',
        type=float,
        dest='drop_prob',
        default=0.5,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------

    args = parser.parse_args()
    print args

    # Read dataset from hdf5 file
    f_dataset = h5py.File(args.input_file, 'r')
    print('Dataset (%s) attributes:' % (args.input_file))
    for key in f_dataset.attrs.keys():
        print('%s : %s' % (key, str(f_dataset.attrs[key])))

    # Create a model (choosen via argument passing)
    if args.model_type == 'lstmcrf':
        m = SimpleLstmCrfModel(
            f_dataset['training'],
            f_dataset['validation'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type
        )
    elif args.model_type == 'lstm':
        m = SimpleLstmModel(
            f_dataset['training'],
            f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type
        )
    elif args.model_type == 'crf':
        m = SimpleCrfModel(
            f_dataset['training'],
            f_dataset['validation'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            optimizer_type=args.optimizer_type
        )
    else:
        raise NotImplementedError('Please specify a valid model (-M <model_type>).')

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    m.run()
    # -----------------------------------------------

    f_dataset.close()
