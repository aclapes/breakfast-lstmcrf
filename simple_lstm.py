import numpy as np
import tensorflow as tf

from progressbar import ProgressBar

from tensorflow.contrib import rnn

# from reader import read_data_generator
from evaluation import compute_framewise_accuracy

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
        l = lengths[(i * batch_size):((i + 1) * batch_size), :]  # not returned!

        yield (x, y, np.squeeze(l))

class SimpleLstmModel(object):
    def __init__(self, config, input_data, is_training):
        self.config = config
        self.input_data = input_data
        self.is_training = is_training

        num_layers = 2

        no_classes = config['no_classes']
        batch_size = config['batch_size']
        num_words = config['num_words']
        num_features = config['num_features']
        optimizer_type = config['optimizer_type']
        learn_rate = config['learn_rate']
        decay_rate = config['decay_rate']
        hidden_size = config['hidden_size']
        drop_prob = config['drop_prob']
        clip_norm = config['clip_norm']

        decay_steps = self.input_data['video_features'].shape[0] // 32

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[batch_size, num_words, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[batch_size, num_words])
        self.l_batch = tf.placeholder(tf.int32, shape=[batch_size])

        self.state_placeholder = tf.placeholder(tf.float32, shape=[2, 2, batch_size, hidden_size])

        classweights = tf.expand_dims(tf.constant(config['class_weights']), axis=0)

        x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        if is_training:
            x_batch = tf.nn.dropout(x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

        def attn_cell():
            cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
            if is_training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - drop_prob)
            return cell

        cells_fw = [attn_cell() for _ in range(num_layers)]
        cells_bw = [attn_cell() for _ in range(num_layers)]

        self.initial_state_fw = [cell.zero_state(batch_size, dtype=np.float32) for cell in cells_fw]
        self.initial_state_bw = [cell.zero_state(batch_size, dtype=np.float32) for cell in cells_bw]
        # self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])

        rnn_outputs, final_state_fw, final_state_bw  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            x_batch,
            dtype=tf.float32,
            initial_states_fw=self.initial_state_fw,
            initial_states_bw=self.initial_state_bw,
            sequence_length=self.l_batch
        )
        # self.final_state = (final_state_fw, final_state_bw)

        # rnn_outputs = tf.concat(rnn_outputs, 2)

        # rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
        #     cell,
        #     x_batch,
        #     dtype=tf.float32,
        #     initial_state=self.initial_state,  # statefull rnn
        #     sequence_length=self.l_batch  # do not process padded parts
        # )


        matricied_x = tf.reshape(rnn_outputs, [-1, 2*hidden_size])
        softmax_w = tf.get_variable('softmax_w', [2*hidden_size, no_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b

        # Class weighting
        # See: https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow#answer-38912982
        # ---
        y_onehot = tf.one_hot(self.y_batch, no_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_onehot_matricied = tf.reshape(y_onehot, [-1, no_classes])
        weight_per_label = tf.transpose(tf.matmul(y_onehot_matricied, tf.transpose(classweights)))
        xent = tf.multiply(weight_per_label,
                      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot_matricied))
        # ---

        # reshape to sequence format and mask padded timesteps
        xent = tf.reshape(xent, [-1,num_words])
        mask = tf.sequence_mask(self.l_batch)
        masked_xent = tf.boolean_mask(xent, mask)

        # compute loss and framewise predictions
        self.loss = tf.reduce_mean(masked_xent)
        self.predictions = tf.argmax(tf.reshape(logits, [-1,num_words,no_classes]), 2)

        if not is_training:
            return

        global_step = tf.Variable(0, trainable=False)
        # self.curr_learn_rate = tf.train.inverse_time_decay(learn_rate,
        #                                                    global_step,
        #                                                    decay_steps=decay_steps,
        #                                                    decay_rate=decay_rate,
        #                                                    staircase=True)
        boundaries = (np.array([1, 100, 1000], dtype=np.int32) * batch_size).tolist()
        values = [1e-1, 1e-2, 1e-3, 1e-4]
        self.curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)

        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, tvars), global_step=global_step)

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

        # state = session.run(self.initial_state)
        state = np.zeros((2, 2, self.config['batch_size'], self.config['hidden_size']), dtype=np.float32)

        fetches = {
            'cost': self.loss,
            # 'final_state': self.final_state,
            'predictions': self.predictions,
        }
        if self.is_training:
            fetches['train_op'] = self.train_op
            fetches['curr_learn_rate'] = self.curr_learn_rate
            # fetches['grads'] = self.grads

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = self.reader.next()

            # c, h = self.initial_state
            # feed_dict[c] = state.c
            # feed_dict[h] = state.h
            feed_dict = {self.x_batch: batch[0], self.y_batch: batch[1], self.l_batch: batch[2]}
                         # self.state_placeholder: state}

            vals = session.run(fetches=fetches, feed_dict=feed_dict)

            # print vals['final_state'].h[0,:3]
            # state = vals['final_state']
            if self.is_training:
                print vals['curr_learn_rate']

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
                 class_weights,
                 batch_size,
                 learn_rate,
                 decay_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam',
                 clip_norm=1.0):

        self.num_epochs = num_epochs

        config = dict(
            no_classes = no_classes,
            class_weights = class_weights,
            batch_size = batch_size,
            num_words = train['video_features'].shape[1],
            num_features = train['video_features'].shape[2],
            hidden_size = hidden_size,
            drop_prob = drop_prob,
            optimizer_type = optimizer_type,
            clip_norm = clip_norm,
            learn_rate = learn_rate,
            decay_rate = decay_rate
        )

        test_config = config.copy()
        test_config['batch_size'] = 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-0.1,
                                                        0.1)

            with tf.name_scope('Train'):
                with tf.variable_scope('Model', reuse=False, initializer=initializer): #, initializer=initializer):
                    self.train_model = SimpleLstmModel(config=config, input_data=train, is_training=True)
            with tf.name_scope('Validation'):
                with tf.variable_scope('Model', reuse=True, initializer=initializer):
                    self.val_model = SimpleLstmModel(config=config, input_data=val, is_training=False)
            # with tf.name_scope('Test'):
            #     with tf.variable_scope('Model', reuse=True, initializer=initializer):
            #         self.te_model = SimpleLstmModel(config=test_config, input_data=te, is_training=False)

            self.init_op = tf.global_variables_initializer()


    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e + 1, self.num_epochs))
                train_eval = self.train_model.run_epoch(session)
                # print('TRAIN (loss/acc): %.4f/%.2f%%' % (train_eval[0], train_eval[1]))
                val_eval = self.val_model.run_epoch(session)
                print(
                    'TRAIN (loss/acc): %.4f/%.2f%%, VAL (loss/acc): %.4f/%.2f%%' % (
                        train_eval[0], train_eval[1], val_eval[0], val_eval[1]
                    )
                )
            # te_eval = self.te_model.run_epoch(session)
            #
            # print(
            #     'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' % (
            #         train_eval[1], val_eval[1], te_eval[1])
            # )