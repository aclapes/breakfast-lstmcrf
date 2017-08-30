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
        decay_rate = config['decay_rate']
        learn_rate = config['learn_rate']
        clip_norm = config['clip_norm']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[None, num_words, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[None, num_words])
        self.l_batch = tf.placeholder(tf.int32, shape=[None])

        x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        if is_training:
            x_batch = tf.nn.dropout(x_batch, keep_prob=1.0)  # TODO: experiment with this dropout

        # Compute unary scores from a linear layer.
        matricied_x = tf.reshape(x_batch, [-1, num_features])

        hidden_w = tf.get_variable('hidden_w', [num_features, config['hidden_size']], dtype=tf.float32)
        hidden_b = tf.get_variable('hidden_b', [config['hidden_size']], dtype=tf.float32, initializer=tf.zeros_initializer())
        hidden_activations = tf.matmul(matricied_x, hidden_w) + hidden_b
        hidden_activations = tf.layers.dropout(hidden_activations, rate=config['drop_prob'], training=config['drop_prob'])

        softmax_w = tf.get_variable('softmax_w', [config['hidden_size'], no_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden_activations, softmax_w) + softmax_b

        normalized_logits = tf.nn.softmax(logits)
        unary_scores = tf.reshape(
            normalized_logits, [batch_size, num_words, no_classes]
        )

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = crf.crf_log_likelihood(
            unary_scores, self.y_batch, self.l_batch)
        # Add a training op to tune the parameters.

        self.decoding, _ = crf.crf_decode(unary_scores, transition_params, self.l_batch)


        self.loss = tf.reduce_mean(-log_likelihood)

        if not is_training:
            return

        global_step = tf.Variable(0, trainable=False)
        boundaries = (np.array([1, 100, 1000], dtype=np.int32) * batch_size).tolist()
        values = [1e-1, 1e-2, 1e-3, 1e-4]
        curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=curr_learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)


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
                feed_dict={self.x_batch: batch[0], self.y_batch: batch[1], self.l_batch: batch[2]}
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
                 decay_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam',
                 clip_norm=1.0):

        self.num_epochs = num_epochs

        config = dict(
            no_classes = no_classes,
            batch_size = batch_size,
            num_words = train['video_features'].shape[1],
            num_features = train['video_features'].shape[2],
            optimizer_type = optimizer_type,
            hidden_size = hidden_size,
            drop_prob = drop_prob,
            decay_rate = decay_rate,
            learn_rate = learn_rate,
            clip_norm = clip_norm
        )

        test_config = config.copy()
        test_config['batch_size'] = 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', reuse=False, initializer=initializer):
                self.train_model = SimpleCrfModel(config=config, input_data=train, is_training=True)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.val_model = SimpleCrfModel(config=config, input_data=val, is_training=False)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.te_model = SimpleCrfModel(config=test_config, input_data=te, is_training=False)

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