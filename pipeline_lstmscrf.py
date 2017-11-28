import numpy as np
import tensorflow as tf
from progressbar import ProgressBar
from tensorflow.contrib import rnn
from os import path, makedirs

import src.scrf as scrf  # master's version of tf.contrib.crf
from src.evaluation import compute_framewise_accuracy, compute_classwise_accuracy
from src.preprocessing import compute_class_weights
from src.reader import read_data_generator
import exceptions
# from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

from src import scrf

def outputs2transitions(y):
    transitions = [y[0]]  # dummy class id is 0
    for i in range(1,len(y)):
        if y[i] != y[i-1]:
            transitions.append(y[i])

    return transitions

class SimpleLstmScrfModel(object):
    def __init__(self, config, input_data, test_subset, summaries_dir, is_training=False):
        self.config = config

        self.dataset = input_data['dataset']
        self.lengths = input_data['lengths']
        test_mask = (np.array(input_data['subsets']) == test_subset)
        indices = np.where(test_mask == (False if is_training else True))[0]
        self.indices_sb = indices
        self.is_training = is_training

        num_features = config['num_features']
        num_classes = config['num_classes']
        batch_size = config['batch_size']
        optimizer_type = config['optimizer_type']
        learn_rate = config['learn_rate']
        decay_rate = config['decay_rate']
        hidden_state_size = config['hidden_size']
        drop_prob = config['drop_prob']
        clip_norm = config['clip_norm']

        self.class_weights = config['class_weights']

        # DEBUG
        transition_matrix = np.zeros_like(input_data['transition_matrix'], dtype=input_data['transition_matrix'])


        # GRAPH CONSTRUCTION

        # -----------------------------------
        # Input pipeline
        # -----------------------------------

        def batch_generator():
            """
            In-line function to generate data batches from an iterator.
            :return: a 3-length tuple: (video_features, outputs, and lengths)
            """
            perm = np.random.permutation(indices.shape[0])
            n = len(perm)
            num_batches = int(np.ceil(float(n)/self.config['batch_size'])) # discard last batch

            for b in range(num_batches):
                ptr_b = b * self.config['batch_size']
                batch_size = min(self.config['batch_size'], n-ptr_b)

                video_features_b = []
                outputs_b = []
                lengths_b = [self.lengths[indices[perm[ptr_b+i]]] for i in range(batch_size)]
                transitions_b = []
                # indices_b = [indices[perm[ptr_b+i]] for i in range(batch_size)]  # debug

                maxlen_b = np.max(lengths_b)
                maxtrans_b = np.max([len(self.dataset[indices[perm[ptr_b+i]]]['transitions']) for i in range(batch_size)])
                for i in range(batch_size):
                    idx = indices[perm[ptr_b+i]]
                    instance = self.dataset[idx]
                    length = self.lengths[idx]

                    video_features = np.pad(np.reshape(instance['video_features'], [self.lengths[idx], -1]),
                                            ((0, maxlen_b-length), (0, 0)),
                                            'constant', constant_values=0)
                    outputs = np.pad(instance['outputs'], (0, maxlen_b-length), 'constant', constant_values=-1)
                    transitions = np.pad(instance['transitions'], (0, maxtrans_b-len(instance['transitions'])), 'constant', constant_values=-1)

                    video_features_b.append(video_features)
                    outputs_b.append(outputs)
                    transitions_b.append(transitions)

                yield (np.array(video_features_b), np.array(outputs_b), np.array(lengths_b), np.array(transitions_b))

        # Reading the data
        # <-- (ONLY DEBUG, not working on full run)
        # g = batch_generator()
        # for i in range(10):
        #     x_batch, y_batch, l_batch, t_batch = g.next()
        # ---
        with tf.device('/cpu:0'):
            self.iterator = (
                tf.data.Dataset.from_generator(
                    batch_generator,
                    (tf.float32, tf.int32, tf.int32, tf.int32),
                    (
                        tf.TensorShape([None, None, num_features]),
                        tf.TensorShape([None, None]),
                        tf.TensorShape([None]),
                        tf.TensorShape([None, None])
                    )
                )
                .prefetch(2)
                .repeat()
            ).make_one_shot_iterator()

            x_batch, y_batch, l_batch, t_batch = self.iterator.get_next()
        # -->

        # -----------------------------------
        # Scrf
        # -----------------------------------

        cell_fw = rnn.BasicLSTMCell(hidden_state_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)
        cell_bw = rnn.BasicLSTMCell(hidden_state_size, forget_bias=1.0, state_is_tuple=True,
                                 reuse=tf.get_variable_scope().reuse)

        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                                state_keep_prob=(0.5 if is_training else 1),
                                                output_keep_prob=(0.2 if is_training else 1),
                                                variational_recurrent=True,
                                                input_size=tf.TensorShape([None, None, num_features]),
                                                dtype=tf.float32)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,
                                                state_keep_prob=(0.5 if is_training else 1),
                                                output_keep_prob=(0.2 if is_training else 1),
                                                variational_recurrent=True,
                                                input_size=tf.TensorShape([None, None, num_features]),
                                                dtype=tf.float32)

        self.initial_state_fw = cell_fw.zero_state(tf.shape(x_batch)[0], dtype=np.float32)
        self.initial_state_bw = cell_bw.zero_state(tf.shape(x_batch)[0], dtype=np.float32)
        # self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])

        (rnn_outputs_fw, rnn_outputs_fw), self.final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            x_batch,
            dtype=tf.float32,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            sequence_length=l_batch  # do not process padded parts
        )
        rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_fw], axis=2)

        matricied_x = tf.reshape(rnn_outputs, [-1, 2*hidden_state_size])
        # Alternatives:
        # (1) Use a hidden-layer MLP
        # <--
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
        # (2) Only output layer MLP
        softmax_w = tf.get_variable('softmax_w', [2*hidden_state_size, num_classes-1], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [num_classes-1], dtype=tf.float32)
        output = tf.matmul(matricied_x, softmax_w) + softmax_b

        output = tf.nn.softmax(output, dim=-1)

        x_scrf_input = tf.reshape(output, [-1, tf.shape(rnn_outputs)[1], num_classes-1])

        with tf.name_scope('lstmcost'):
            mask = tf.sequence_mask(l_batch, tf.shape(x_scrf_input)[1], dtype=tf.int32)
            y_seq = (y_batch-1) * mask
            # compute loss and framewise predictions (uses softmax on logits internally)
            seq_loss = tf.contrib.seq2seq.sequence_loss(
                x_scrf_input,
                y_seq,
                tf.cast(mask, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=False
            )
            self.lstm_loss = tf.reduce_mean(seq_loss)
            # tf.summary.scalar('loss', self.loss)

        self.lstm_predictions = tf.argmax(x_scrf_input, axis=-1) + 1  # make it coincide with the indexing of labels in y_batch

        log_likelihood, _, _ = scrf.scrf_log_likelihood(
            tf.cast(x_scrf_input, dtype=tf.float32),
            orderings=t_batch,
            lengths=l_batch,
            transition_matrix=tf.cast(transition_matrix, dtype=tf.float32)
        )

        self.viterbi, self.backpointers = scrf.scrf_viterbi_dp(x_scrf_input, l_batch, transition_matrix)
        self.l_batch = l_batch

        with tf.name_scope('lstmscrfcost'):
            # compute loss and framewise predictions
            self.loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar('loss', self.loss)

        # with tf.name_scope('evaluation'):
        #     # self.predictions, _ = Scrf.Scrf_decode(unary_scores, transition_params, l_batch)
        #     self.predictions = None # Fix
        #     self.y_batch = y_batch
        #     self.l_batch = l_batch
        #     equality = tf.cast(tf.equal(self.predictions, y_batch), tf.float32)
        #     mask = tf.sequence_mask(l_batch, tf.shape(y_batch)[1])
        #     self.acc = tf.reduce_sum(tf.boolean_mask(equality, mask)) / tf.reduce_sum(tf.cast(mask, tf.float32))
        #     tf.summary.scalar('accuracy', self.acc)

        self.global_step = tf.Variable(0, trainable=False)
        # boundaries = (np.array([5,10], dtype=np.int32) * batch_size).tolist()
        # values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        # curr_learn_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, name=None)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)

        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # tvars = tf.trainable_variables()
        # self.grads = tf.gradients(self.loss, tvars)
        # grads_clipped, _ = tf.clip_by_global_norm(self.grads, clip_norm=1.0)
        # self.train_op = self.optimizer.apply_gradients(zip(grads_clipped, tvars), global_step=self.global_step)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.optimizer_lstm = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_lstm = self.optimizer_lstm.minimize(self.lstm_loss, global_step=self.global_step)

        self.y_batch = y_batch

        #
        # # -----------------------------------
        # # Tensorboard's auxiliary stuff
        # # -----------------------------------
        # self.merge_summaries_op = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(summaries_dir, tf.get_default_graph())


    def run_pretrain(self, session, epoch_nr):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        batch_loss = []
        batch_accs = []

        num_batches = int(np.ceil(len(self.indices_sb)/float(self.config['batch_size'])))
        # session.run(self.init_it)

        bar = ProgressBar(max_value=num_batches)
        for batch_i in range(num_batches):
            bar.update(batch_i)
            try:
                lstm_predictions, outputs, lengths, lstm_loss, _ = session.run([self.lstm_predictions, self.y_batch, self.l_batch, self.lstm_loss, self.train_lstm])
                hits, total = 0, 0
                for i, (output, decoding) in enumerate(zip(outputs, lstm_predictions)):
                    hits  += np.sum(np.equal(output[:lengths[i]], decoding[:lengths[i]]))
                    total += lengths[i]
                acc = float(hits)/total
                batch_loss.append(lstm_loss)
                batch_accs.append(acc)
                # print(' -> loss=%.5f, acc=%.5f' % (lstm_loss, lstm_acc))
                batch_i += 1
            except (tf.errors.OutOfRangeError, exceptions.StopIteration) as e:
                break
        bar.finish()

        return np.nanmean(batch_loss), np.nanmean(batch_accs)


    def run_epoch(self, session, epoch_nr):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        batch_loss = []
        batch_accs = []

        num_batches = int(np.ceil(len(self.indices_sb) / float(self.config['batch_size'])))
        # session.run(self.init_it)

        print('Adjust lstmcrf')
        bar = ProgressBar(max_value=num_batches)
        for batch_i in range(num_batches):
            bar.update(batch_i)

            try:
                # v = session.run([self.x_batch, self.t_final, self.alphas_final])
                # v = session.run([self.p_batch, self.end_inds_batch, self.betas_final, self.l_batch])
                # v = session.run([self.x_batch_reord, self.p_batch, self.betas_final ])
                # v = session.run([self.x_batch_reord, self.p_batch], {self.transitions_placeholder : self.transition_matrix})
                # v = session.run([self.alphas_final])
                # v = session.run([self.output, self.l_batch, self.end_inds_batch, self.alphas_final, self.betas_final, self.betas_prefinal, self.train_op, self.grads, self.loss])
                # print(' -> loss=%.5f, acc(mof)=%2.2f%%' % (vals['loss'], 100.0 * vals['acc']))
                # v = session.run([self.x_itgl, self.y_batch,self.l_batch,self.t_batch])
                # v = session.run([self.x_batch_reord, self.p_batch,self.end_inds_batch])
                # v = session.run([self.betas_prefinal, self.betas_final, self.end_inds_batch])
                viterbi_matrix, backpointers, lengths, y_batch, lstm_predictions, loss, _ = session.run([self.viterbi, self.backpointers, self.l_batch, self.y_batch, self.lstm_predictions, self.loss, self.train_op])
                decodings = scrf.decode_viterbi_dp(viterbi_matrix, backpointers, lengths)

                print
                hits, total = 0, 0
                for i, (y, decoding) in enumerate(zip(y_batch, decodings)):
                    hits  += np.sum(np.equal(y[:lengths[i]], decoding[:lengths[i]]))
                    total += lengths[i]
                    print outputs2transitions(y[:lengths[i]]), " -> ", \
                        outputs2transitions(decoding[:lengths[i]]), " -> ", \
                        outputs2transitions(lstm_predictions[i][:lengths[i]])
                acc = float(hits)/total
                print

                print(' -> loss=%.5f, acc=%.5f' % (loss, acc))
                print
                batch_loss.append(loss)
                batch_accs.append(acc)

                # if np.isnan(loss_ret):
                #     error = 1
                # self.writer.add_summary(vals['summaries'], global_step=epoch_nr * num_batches + batch_i)
                # batch_loss.append(loss_ret)

                batch_i += 1
            except (tf.errors.OutOfRangeError, exceptions.StopIteration) as e:
                break
        bar.finish()

        return (np.nanmean(batch_loss), np.nanmean(batch_accs)) #, 100*(hit_classes/true_classes)


class SimpleLstmScrfPipeline(object):
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

        num_features = len(input_data['dataset'][0]['video_features']) / (input_data['lengths'][0])
        assert num_features == input_data.attrs['num_features']

        config = dict(
            batch_size = batch_size,
            num_features = num_features,
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


        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('Train'):
                with tf.variable_scope('Model'): #, initializer=tf.random_uniform_initializer(-0.1, 0.1)):
                    self.train_model = SimpleLstmScrfModel(config=config, input_data=input_data,
                                                          test_subset=test_subset,
                                                          summaries_dir=path.join(logging_path, 'train'),
                                                          is_training=True)
            # with tf.name_scope('Validation'):
            #     with tf.variable_scope('Model', reuse=True):
            #         self.val_model = SimpleLstmScrfModel(config=val_config, input_data=input_data,
            #                                             test_subset=test_subset,
            #                                             summaries_dir=path.join(logging_path, 'validation'),
            #                                             is_training=False)
            # with tf.name_scope('Test'):
            #     with tf.variable_scope('Model', reuse=True):
            #         self.te_model = SimpleLstmScrfModel(config=test_config, input_data=input_data,
            #                                            test_subset=test_subset,
            #                                            summaries_dir=path.join(logging_path, 'test'),
            #                                            is_training=False)

            self.init_op = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            # self.saver = tf.train.Saver()


    def run(self, gpu_options, restore_from_ckpt=None):
        np.set_printoptions(precision=2,linewidth=200)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # if restore_from_ckpt is None:
            #     restore_from_ckpt = tf.train.latest_checkpoint(self.output_models_path)
            # try:
            #     self.saver.restore(
            #         session,
            #         restore_from_ckpt
            #     )
            #     print 'Model restored'
            # except ValueError, e:
            #     pass

            session.run(self.init_op)
            # session.run(self.train_model.init_it)
            # session.run(self.val_model.init_it)
            # session.run(self.te_model.init_it)

            # num_pretrain = 10
            # for e in range(num_pretrain):
            #     print('Pretrain: %d/%d' % (e + 1, num_pretrain))
            #     loss_train, mof_train = self.train_model.run_pretrain(session, e)
            #     print('[Pretraining epoch] loss=%.5f, acc=%2.2f%%' % (loss_train, 100.0 * mof_train))

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e+1, self.num_epochs))

                # Train step
                loss_train, mof_train = self.train_model.run_epoch(session, e)
                print('[Training epoch] loss=%.5f, acc=%2.2f%%' % (loss_train, 100.0 * mof_train))

                # Validation step
                # loss_val, mof_val = self.val_model.run_epoch(session, e)
                # print('[Validation epoch] loss=%.5f, acc=%2.2f%%' % (loss_val, 100.0 * mof_val))
                #
                # if e+1 in set([10, 50, 100, 150, 200, 250, self.num_epochs]):
                #     # Save the model
                #     # self.saver.save(
                #     #     session,
                #     #     path.join(self.output_models_path, 'ckpt'),
                #     #     global_step=e,
                #     # )
                #     # print 'Model saved'
                #
                #     loss_te, mof_te = self.te_model.run_epoch(session, e)
                #     print('[Testing epoch] loss=%.5f, acc=%2.2f%%' % (loss_te, 100.0 * mof_te))

