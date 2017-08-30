import numpy as np
import tensorflow as tf

from progressbar import ProgressBar

from tensorflow.contrib import rnn
import src.crf as crf  # master's version of tf.contrib.crf

from reader import read_data_generator
from evaluation import compute_framewise_accuracy


class SimpleLstmcrfModel(object):
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
        decay_rate = config['decay_rate']
        hidden_size = config['hidden_size']
        drop_prob = config['drop_prob']
        clip_norm = config['clip_norm']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[None, num_words, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[None, num_words])
        self.l_batch = tf.placeholder(tf.int32, shape=[None])

        # self.state_placeholder = tf.placeholder(tf.float32, shape=[2, batch_size, hidden_size])

        x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        if is_training:
            x_batch = tf.nn.dropout(x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

        cell_fw = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
        cell_bw = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
        if is_training:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                                    output_keep_prob=1-drop_prob,
                                                    variational_recurrent=True,
                                                    dtype=tf.float32)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,
                                                    output_keep_prob=1-drop_prob,
                                                    variational_recurrent=True,
                                                    dtype=tf.float32)

        self.initial_state_fw = cell_fw.zero_state(tf.shape(self.x_batch)[0], dtype=np.float32)
        self.initial_state_bw = cell_bw.zero_state(tf.shape(self.x_batch)[0], dtype=np.float32)
        # self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])

        rnn_outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            x_batch,
            dtype=tf.float32,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            sequence_length=self.l_batch  # do not process padded parts
        )

        rnn_outputs = tf.concat(rnn_outputs, axis=2)

        matricied_x = tf.reshape(rnn_outputs, [-1, 2*hidden_size])
        softmax_w = tf.get_variable('softmax_w', [2*hidden_size, no_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b

        unary_scores = tf.reshape(logits, [-1, num_words, no_classes])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = crf.crf_log_likelihood(
            unary_scores, self.y_batch, self.l_batch)

        # compute loss and framewise predictions
        self.loss = tf.reduce_mean(-log_likelihood)
        self.predictions, _ = crf.crf_decode(unary_scores, transition_params, self.l_batch)

        if not is_training:
            return

        global_step = tf.Variable(0, trainable=False)
        boundaries = (np.array([100, 1000], dtype=np.int32) * batch_size).tolist()
        values = [1e-2, 1e-3, 1e-4]
        curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=curr_learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)

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
                                          self.input_data['lengths'][:, -1],
                                          batch_size=self.config['batch_size'])

        num_instances = self.input_data['video_features'].shape[0]
        num_batches = int(np.ceil(num_instances / float(self.config['batch_size'])))
        batch_loss = [None] * num_batches
        batch_accs = [None] * num_batches

        # state = session.run(self.initial_state)
        # state = np.zeros((2, self.config['batch_size'], self.config['hidden_size']), dtype=np.float32)

        fetches = {
            'loss': self.loss,
            # 'final_state': self.final_state,
            'predictions': self.predictions,
        }
        if self.is_training:
            fetches['train_op'] = self.train_op
            # fetches['curr_learn_rate'] = self.curr_learn_rate
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
            # if self.is_training:
                # print vals['curr_learn_rate']

            batch_loss[b] = vals['loss']
            batch_accs[b] = compute_framewise_accuracy(vals['predictions'], batch[1], batch[2])
            progbar.update(b)
        progbar.finish()

        return np.mean(batch_loss), np.mean(batch_accs)


class SimpleLstmcrfPipeline(object):
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
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            with tf.name_scope('Train'):
                with tf.variable_scope('Model', reuse=False, initializer=initializer): #, initializer=initializer):
                    self.train_model = SimpleLstmcrfModel(config=config, input_data=train, is_training=True)
            with tf.name_scope('Validation'):
                with tf.variable_scope('Model', reuse=True, initializer=initializer):
                    self.val_model = SimpleLstmcrfModel(config=config, input_data=val, is_training=False)
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True, initializer=initializer):
                    self.te_model = SimpleLstmcrfModel(config=test_config, input_data=te, is_training=False)

            self.init_op = tf.global_variables_initializer()


    def run(self, gpu_options):
        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(self.init_op)

            train_evals = [None] * self.num_epochs
            val_evals = [None] * self.num_epochs

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e + 1, self.num_epochs))
                train_evals[e] = self.train_model.run_epoch(session)
                val_evals[e] = self.val_model.run_epoch(session)
                print(
                    'TRAIN (loss/acc): %.4f/%.2f%%, VAL (loss/acc): %.4f/%.2f%%' % (
                        train_evals[e][0], train_evals[e][1], val_evals[e][0], val_evals[e][1]
                    )
                )
                if e in [5, 10, 50, 100, 500, 1000, 2000, 10000, 20000]:  # see progress (not choosing based on this!)
                    _, te_acc = self.te_model.run_epoch(session)
                    print('TE (acc): %.2f%%' % (te_acc))

            tr_acc = np.mean([acc for _,acc in train_evals])
            val_acc = np.mean([acc for _,acc in val_evals])

            _, te_acc = self.te_model.run_epoch(session)
            print(
                'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' % (
                    tr_acc, val_acc, te_acc)
            )