import numpy as np
import tensorflow as tf

from progressbar import ProgressBar

from tensorflow.contrib import rnn

from reader import read_data_generator
from evaluation import compute_framewise_accuracy


class SimpleLstmModel(object):
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
        hidden_size = config['hidden_size']
        drop_prob = config['drop_prob']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[batch_size, num_words, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[batch_size, num_words])
        self.w_batch = tf.placeholder(tf.float32, shape=[batch_size, num_words])
        # get sequences length from binary mask of valid timesteps
        lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

        x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        if is_training:
            x_batch = tf.nn.dropout(x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

        cell = rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
        if is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_prob)

        self.initial_state = cell.zero_state(batch_size, dtype=np.float32)

        rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
            cell,
            x_batch,
            dtype=tf.float32,
            initial_state=self.initial_state,  # statefull rnn
            sequence_length=lengths_batch  # do not process padded parts
        )


        # Compute unary scores from a linear layer.
        matricied_x = tf.reshape(rnn_outputs, [-1, hidden_size])
        softmax_w = tf.get_variable('softmax_w', [hidden_size, no_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b

        normalized_logits = tf.nn.softmax(logits)
        normalized_logits = tf.reshape(
            normalized_logits, [batch_size, num_words, no_classes]
        )

        self.predictions = tf.argmax(normalized_logits, 2)

        loss = tf.contrib.seq2seq.sequence_loss(
            normalized_logits,
            self.y_batch,
            self.w_batch,
            average_across_timesteps=False,
            average_across_batch=True
        )
        self.cost = tf.reduce_sum(loss)

        if not is_training:
            return

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)

        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5.0)
        self.train_op = optimizer.apply_gradients(
            zip(self.grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

    def run_epoch(self, session):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        self.reader = read_data_generator(self.input_data['video_features'],
                                          self.input_data['outputs'],
                                          self.input_data['lengths'],
                                          batch_size=self.config['batch_size'])

        num_batches = self.input_data['video_features'].shape[0] // self.config['batch_size']
        batch_loss = [None] * num_batches
        batch_accs = [None] * num_batches

        state = session.run(self.initial_state)

        fetches = {
            'cost': self.cost,
            'final_state': self.final_state,
            'predictions': self.predictions,
        }
        if self.is_training:
            fetches['train_op'] = self.train_op
            fetches['grads'] = self.grads

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = self.reader.next()

            feed_dict = {self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2]}

            c, h = self.initial_state
            feed_dict[c] = state.c
            feed_dict[h] = state.h

            vals = session.run(fetches=fetches, feed_dict=feed_dict)

            print vals['final_state'].h[0,:3]
            state = vals['final_state']

            batch_loss[b] = vals['cost']
            batch_accs[b] = compute_framewise_accuracy(vals['predictions'], batch[1], batch[2])

            progbar.update(b)
        progbar.finish()

        return np.mean(batch_loss), np.mean(batch_accs)


class SimpleLstmPipeline(object):
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

        self.num_epochs = num_epochs

        config = dict(
            no_classes = no_classes,
            batch_size = batch_size,
            num_words = train['video_features'].shape[1],
            num_features = train['video_features'].shape[2],
            hidden_size = hidden_size,
            drop_prob = drop_prob,
            optimizer_type = optimizer_type,
            learn_rate = learn_rate
        )

        test_config = config.copy()
        test_config['batch_size'] = 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            # initializer = tf.random_uniform_initializer(-0.05, 0.05)
            with tf.variable_scope('Model', reuse=False):#, initializer=initializer):
                self.train_model = SimpleLstmModel(config=config, input_data=train, is_training=True)

            with tf.variable_scope('Model', reuse=True):#, initializer=initializer):
                self.val_model = SimpleLstmModel(config=config, input_data=val, is_training=False)

            with tf.variable_scope('Model', reuse=True):#, initializer=initializer):
                self.te_model = SimpleLstmModel(config=test_config, input_data=te, is_training=False)

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