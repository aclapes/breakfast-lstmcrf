import numpy as np
import tensorflow as tf
from progressbar import ProgressBar
from tensorflow.contrib import rnn
from os import path, makedirs

import src.crf as crf  # master's version of tf.contrib.crf
from src.evaluation import compute_framewise_accuracy, compute_classwise_accuracy
from src.preprocessing import compute_class_weights
from src.reader import read_data_generator
import exceptions

class SimpleLstmcrfModel(object):
    def __init__(self, config, input_data, test_subset, summaries_dir, is_training=False):
        self.config = config

        self.dataset = input_data['dataset']
        self.lengths = input_data['lengths']
        test_mask = (np.array(input_data['subsets']) == test_subset)
        indices = np.where(test_mask == (False if is_training else True))[0]
        self.indices_sb = indices
        self.is_training = is_training

        num_classes = config['num_classes']
        batch_size = config['batch_size']
        optimizer_type = config['optimizer_type']
        learn_rate = config['learn_rate']
        decay_rate = config['decay_rate']
        hidden_state_size = config['hidden_size']
        drop_prob = config['drop_prob']
        clip_norm = config['clip_norm']

        self.class_weights = config['class_weights']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        # self.x_batch = tf.placeholder(tf.float32, shape=[None, num_words, num_features])
        # self.y_batch = tf.placeholder(tf.int32, shape=[None, num_words])
        # self.l_batch = tf.placeholder(tf.int32, shape=[None])

        # self.state_placeholder = tf.placeholder(tf.float32, shape=[2, batch_size, hidden_size])

        # x_batch = self.x_batch

        # def gen():
        #     """ A simple data iterator """
        #     n =  self.data_indices.shape[0]
        #     perm = np.random.permutation(n)
        #     end = self.config['batch_size']*(n//self.config['batch_size']) # discard last batch
        #     perm = perm[:end]
        #     for i in perm:
        #         idx = self.data_indices[i]
        #         video_features = np.reshape(self.input_data[idx]['video_features'], [len(self.input_data[idx]['outputs']),-1])
        #         outputs = self.input_data[idx]['outputs']
        #         yield (video_features, outputs)

        def gen():
            """ A simple data iterator """
            perm = np.random.permutation(indices.shape[0])
            n = len(perm)
            num_batches = int(np.ceil(float(n)/self.config['batch_size'])) # discard last batch

            for b in range(num_batches):
                ptr_b = b * self.config['batch_size']
                batch_size = min(self.config['batch_size'], n-ptr_b)

                video_features_b = []
                outputs_b = []
                lengths_b = [self.lengths[indices[perm[ptr_b+i]]] for i in range(batch_size)]
                indices_b = [indices[perm[ptr_b+i]] for i in range(batch_size)]
                maxlen_b = np.max(lengths_b)
                for i in range(batch_size):
                    idx = indices[perm[ptr_b+i]]
                    instance = self.dataset[idx]
                    length = self.lengths[idx]

                    video_features = np.pad(np.reshape(instance['video_features'], [self.lengths[idx], -1]),
                                            ((0, maxlen_b-length), (0, 0)),
                                            'constant', constant_values=0)

                    outputs = np.pad(instance['outputs'], (0, maxlen_b-length), 'constant', constant_values=0)

                    video_features_b.append(video_features)
                    outputs_b.append(outputs)

                yield (np.array(video_features_b), np.array(outputs_b), lengths_b, indices_b)

        # <--- DEBUG (gen function)
        # g = gen()
        # for i in range(10):
        #     x_batch, y_batch, l_batch = g.next()
        # DEBUG --->
        with tf.device('/cpu:0'):

            self.iterator = (
                tf.data.Dataset.from_generator(
                    gen,
                    (tf.float32, tf.int32, tf.int32, tf.int32),
                    (
                        tf.TensorShape([None, None, 64]),
                        tf.TensorShape([None, None]),
                        tf.TensorShape([None, ]),
                        tf.TensorShape([None])
                    )
                )
                .prefetch(2)
                .repeat()
            ).make_one_shot_iterator()

            # self.init_it = self.iterator.initializer
            x_batch, y_batch, l_batch, self.i_batch = self.iterator.get_next()

        cell_fw = rnn.BasicLSTMCell(hidden_state_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
        cell_bw = rnn.BasicLSTMCell(hidden_state_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)

        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                                state_keep_prob=(0.5 if is_training else 1),
                                                output_keep_prob=(0.2 if is_training else 1),
                                                variational_recurrent=True,
                                                input_size=tf.TensorShape([None,None,64]),
                                                dtype=tf.float32)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,
                                                state_keep_prob=(0.5 if is_training else 1),
                                                output_keep_prob=(0.2 if is_training else 1),
                                                variational_recurrent=True,
                                                input_size=tf.TensorShape([None, None, 64]),
                                                dtype=tf.float32)

        self.initial_state_fw = cell_fw.zero_state(tf.shape(x_batch)[0], dtype=np.float32)
        self.initial_state_bw = cell_bw.zero_state(tf.shape(x_batch)[0], dtype=np.float32)
        # self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])

        rnn_outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            x_batch,
            dtype=tf.float32,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            sequence_length=l_batch  # do not process padded parts
        )

        rnn_outputs = tf.concat(rnn_outputs, axis=2)
        # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=(0.2 if is_training else 1))
        # rnn_outputs = tf.concat([x_batch, rnn_outputs], axis=2)

        matricied_x = tf.reshape(rnn_outputs, [-1, 2*hidden_state_size])
        # Alternatives:
        # (1)
        # <---
        # hidden_neurons = hidden_state_size
        # hidden_w = tf.get_variable('hidden_w', [2*hidden_state_size, hidden_neurons], dtype=tf.float32)
        # hidden_b = tf.get_variable('hidden_b', [hidden_neurons], dtype=tf.float32, initializer=tf.zeros_initializer())
        # hidden_activations = tf.matmul(matricied_x, hidden_w) + hidden_b
        # if is_training:
        #     hidden_activations = tf.nn.dropout(hidden_activations, keep_prob=0.5)
        #
        # softmax_w = tf.get_variable('softmax_w', [hidden_neurons, num_classes], dtype=tf.float32, regularizer=tf.contrib.layers.l1_regularizer(scale=0.001))
        # softmax_b = tf.get_variable('softmax_b', [num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        # logits = tf.matmul(hidden_activations, softmax_w) + softmax_b
        # ---
        # (2)
        softmax_w = tf.get_variable('softmax_w', [2*hidden_state_size, num_classes], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [num_classes], dtype=tf.float32)
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b
        # ---
        # (3)
        # softmax_w = tf.get_variable('softmax_w', [2*hidden_state_size, 64], dtype=tf.float32)
        # softmax_b = tf.get_variable('softmax_b', [128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # logits = tf.tanh(tf.matmul(matricied_x, softmax_w))
        #
        # softmax_w2 = tf.get_variable('softmax_w2', [64, 64], dtype=tf.float32)
        # # softmax_b2 = tf.get_variable('softmax_b2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # logits2 = tf.tanh(tf.matmul(matricied_x2, softmax_w2))
        # # --->
        #
        # logits = tf.concat([logits,logits2], axis=1) + softmax_b

        unary_params = tf.get_variable("unary_params", shape=[num_classes], dtype=tf.float32)
        unary_scores = tf.reshape(tf.multiply(logits, unary_params), [-1, tf.shape(rnn_outputs)[1], num_classes])
        # unary_scores = tf.reshape(logits, [-1, tf.shape(rnn_outputs)[1], 128])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = crf.crf_log_likelihood(
            unary_scores, y_batch, l_batch)

        with tf.name_scope('cost'):
            # compute loss and framewise predictions
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(-log_likelihood) #+ tf.add_n(reg_losses)
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('evaluation'):
            self.predictions, _ = crf.crf_decode(unary_scores, transition_params, l_batch)
            self.y_batch = y_batch
            self.l_batch = l_batch
            equality = tf.cast(tf.equal(self.predictions, y_batch), tf.float32)
            mask = tf.sequence_mask(l_batch, tf.shape(y_batch)[1])
            self.acc = tf.reduce_sum(tf.boolean_mask(equality, mask)) / tf.reduce_sum(tf.cast(mask, tf.float32))
            tf.summary.scalar('accuracy', self.acc)

        self.global_step = tf.Variable(0, trainable=False)
        boundaries = (np.array([5,10], dtype=np.int32) * batch_size).tolist()
        values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        curr_learn_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, name=None)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=curr_learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)

        # tvars = tf.trainable_variables()
        # self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=clip_norm)
        # self.train_op = self.optimizer.apply_gradients(zip(self.grads, tvars), global_step=global_step)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merge_summaries_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(summaries_dir, tf.get_default_graph())


    def run_epoch(self, session, epoch_nr):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        batch_loss = []
        batch_accs = []

        fetches = {
            'loss': self.loss,
            'acc' : self.acc,
            # 'final_state': self.final_state,
            'predictions': self.predictions,
            'y_batch' : self.y_batch,
            'l_batch' : self.l_batch,
            'i_batch' : self.i_batch,
            'summaries' : self.merge_summaries_op
        }
        if self.is_training:
            fetches['train_op'] = self.train_op
            # fetches['curr_learn_rate'] = self.curr_learn_rate
            # fetches['grads'] = self.grads

        num_batches = int(np.ceil(len(self.indices_sb)/float(self.config['batch_size'])))
        # session.run(self.init_it)

        bar = ProgressBar(max_value=num_batches)
        for batch_i in range(num_batches):
            bar.update(batch_i)

            try:
                vals = session.run(fetches=fetches)
                batch_loss.append(vals['loss'])
                batch_accs.append(vals['acc'])

                print(' -> loss=%.5f, acc(mof)=%2.2f%%' % (vals['loss'], 100.0 * vals['acc']))
                print vals['i_batch']
                # self.writer.add_summary(vals['summaries'], global_step=epoch_nr * num_batches + batch_i)
                batch_i += 1
            except (tf.errors.OutOfRangeError, exceptions.StopIteration) as e:
                print e
                print 'exiting epoch at', batch_i
                break
        bar.finish()

        # session.run(self.init_it)  # reinitialize iterator for next epoch

        mean_loss = np.mean(batch_loss)
        mean_acc = np.mean(batch_accs)

        return (mean_loss, mean_acc) #, 100*(hit_classes/true_classes)


class SimpleLstmcrfPipeline(object):
    def __init__(self,
                 input_data,
                 test_subset,
                 logging_path,
                 output_models_path,
                 batch_size,
                 learn_rate,
                 decay_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam',
                 clip_norm=1.0):

        self.num_epochs = num_epochs
        self.output_models_path = output_models_path

        config = dict(
            batch_size = batch_size,
            # num_words = 1000, #input_data.attrs['max_len'],
            # num_features = 64, #input_data.attrs['num_features'],
            hidden_size = hidden_size,
            drop_prob = drop_prob,
            optimizer_type = optimizer_type,
            clip_norm = clip_norm,
            learn_rate = learn_rate,
            decay_rate = decay_rate
        )

        # try:
        #     class_weights = np.load(class_weights_file)
        # except IOError, e:
        #     class_weights = compute_class_weights(train, config['batch_size'])
        #     np.save(class_weights_file, class_weights)
        class_weights = input_data['class_weights']

        config['class_weights'] = class_weights
        config['num_classes'] = len(class_weights)

        self.sorting = np.argsort(class_weights)  # using class weight criterion

        val_config = config.copy()

        test_config = config.copy()
        test_config['batch_size'] = 1

        # for dir in ['train', 'validation', 'test']:
        #     try:
        #         makedirs(path.join(logging_path, dir))
        #     except:
        #         pass

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('Train'):
                with tf.variable_scope('Model', initializer=tf.random_uniform_initializer(-0.01, 0.01)):
                    self.train_model = SimpleLstmcrfModel(config=config, input_data=input_data,
                                                          test_subset=test_subset,
                                                          summaries_dir=path.join(logging_path, 'train'),
                                                          is_training=True,

                                                          )
            with tf.name_scope('Validation'):
                with tf.variable_scope('Model', reuse=True):
                    self.val_model = SimpleLstmcrfModel(config=val_config, input_data=input_data,
                                                        test_subset=test_subset,
                                                        summaries_dir=path.join(logging_path, 'validation'),
                                                        is_training=False,
                                                        )
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True):
                    self.te_model = SimpleLstmcrfModel(config=test_config, input_data=input_data,
                                                       test_subset=test_subset,
                                                       summaries_dir=path.join(logging_path, 'test'),
                                                       is_training=False,
                                                       )

            self.init_op = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()


    def run(self, gpu_options):
        np.set_printoptions(precision=2,linewidth=200)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            try:
                self.saver.restore(
                    session,
                    tf.train.latest_checkpoint(self.output_models_path)
                )
                print 'Model restored'
            except ValueError, e:
                pass

            session.run(self.init_op)
            # session.run(self.train_model.init_it)
            # session.run(self.val_model.init_it)
            # session.run(self.te_model.init_it)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e+1, self.num_epochs))

                # Train step
                loss_train, mof_train = self.train_model.run_epoch(session, e)
                print('[Training epoch] loss=%.5f, acc=%2.2f%%' % (loss_train, 100.0 * mof_train))

                # Validation step
                loss_val, mof_val = self.val_model.run_epoch(session, e)
                print('[Validation epoch] loss=%.5f, acc=%2.2f%%' % (loss_val, 100.0 * mof_val))

                if e+1 in set([10, 50, 100, 150, 200, self.num_epochs]):
                    # Save the model
                    self.saver.save(
                        session,
                        path.join(self.output_models_path, 'ckpt'),
                        global_step=e,
                        write_meta_graph=(e+1 == self.num_epochs)
                    )
                    print 'Model saved'

            loss_te, mof_te = self.te_model.run_epoch(session, e)
            print('[Testing epoch] loss=%.5f, acc=%2.2f%%' % (loss_te, 100.0 * mof_te))

