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
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

from src import scrf

def outputs2transitions(y):
    '''
    Auxiliary function to transform framewise labels to a list of transitions.
    Used for debugging/result visualization.
    :param y: A list of labels, e.g. [0, 1, 1, 34, 34, 34, 5, ..., 5, 5, 1, 1, 1]
    :return: A list of transitions, e.g. [0, 1, 34, 5, 1]
    '''
    transitions = [y[0]]  # dummy class id is 0
    for i in range(1,len(y)):
        if y[i] != y[i-1]:
            transitions.append(y[i])

    return transitions


class SimpleScrfModel(object):
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
        # ----------------
        # <-- DEBUG (not working on full run, only on first batch)
        # g = batch_generator()
        # x_batch, y_batch, l_batch, t_batch = g.next()
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
                ).prefetch(2).repeat()).make_one_shot_iterator()

            x_batch, y_batch, l_batch, t_batch = self.iterator.get_next()
        # -->

        # -----------------------------------
        # Scrf
        # -----------------------------------

        matricied_x = tf.reshape(x_batch, [-1, num_features])

        hidd_w = tf.get_variable('hidd_w', [num_features, 128], dtype=tf.float32, initializer=tf.random_normal_initializer())
        hidd_b = tf.get_variable('hidd_b', [128], dtype=tf.float32, initializer=tf.zeros_initializer())
        hidd_output = tf.nn.tanh(tf.matmul(matricied_x, hidd_w) + hidd_b)

        proj_w = tf.get_variable('proj_w', [128, num_classes], dtype=tf.float32, initializer=tf.random_normal_initializer())
        proj_b = tf.get_variable('proj_b', [num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.output = tf.matmul(hidd_output, proj_w) + proj_b
        self.output = tf.nn.softmax(self.output)

        x_scrf_input = tf.reshape(self.output, [-1, tf.shape(x_batch)[1], num_classes])
        transition_matrix = input_data['transition_matrix']  # [num_classes+1, num_classes+1]. 0-th index is dummy class

        log_likelihood, _, _ = scrf.scrf_log_likelihood(
            tf.cast(x_scrf_input, dtype=tf.float32),
            orderings=t_batch,
            lengths=l_batch,
            transition_matrix=tf.cast(transition_matrix, dtype=tf.float32)
        )

        self.viterbi_matrix, self.backpointers = scrf.scrf_viterbi_dp(x_scrf_input, l_batch, transition_matrix)

        with tf.name_scope('cost'):
            self.loss = -tf.reduce_mean(log_likelihood)
            tf.summary.scalar('loss', self.loss)

        self.global_step = tf.Variable(0, trainable=False)

        if optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        elif optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # Extra variables we may want to fetch in session's run
        self.y_batch = y_batch
        self.l_batch = l_batch


    def run_epoch(self, session, epoch_nr):
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
                viterbi, backpointers, lengths, outputs, loss, _ = session.run([self.viterbi_matrix, self.backpointers, self.l_batch, self.y_batch, self.loss, self.train_op])
                decodings = scrf.decode_viterbi_dp(viterbi, backpointers, lengths)

                hits, total = 0., 0.
                for output, decoding in zip(outputs, decodings):
                    mask = output > 0
                    hits  += np.sum(np.equal(output[mask], decoding[mask]))
                    total += np.sum(mask)
                acc = hits / total

                print(' -> loss=%.5f, acc=%.5f' % (loss, acc))
                batch_loss.append(loss)
                batch_accs.append(acc)

                batch_i += 1
            except (tf.errors.OutOfRangeError, exceptions.StopIteration) as e:
                break
        bar.finish()

        return (np.nanmean(batch_loss), np.nanmean(batch_accs)) #, 100*(hit_classes/true_classes)


class SimpleScrfPipeline(object):
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
        config['num_classes'] = len(class_weights)-1  # class 0 is a dummy class

        self.sorting = np.argsort(class_weights)  # using class weight criterion

        val_config = config.copy()

        test_config = config.copy()
        test_config['batch_size'] = 1


        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('Train'):
                with tf.variable_scope('Model'): #, initializer=tf.random_uniform_initializer(-0.1, 0.1)):
                    self.train_model = SimpleScrfModel(config=config, input_data=input_data,
                                                          test_subset=test_subset,
                                                          summaries_dir=path.join(logging_path, 'train'),
                                                          is_training=True)
            # with tf.name_scope('Validation'):
            #     with tf.variable_scope('Model', reuse=True):
            #         self.val_model = SimpleScrfModel(config=val_config, input_data=input_data,
            #                                             test_subset=test_subset,
            #                                             summaries_dir=path.join(logging_path, 'validation'),
            #                                             is_training=False)
            # with tf.name_scope('Test'):
            #     with tf.variable_scope('Model', reuse=True):
            #         self.te_model = SimpleScrfModel(config=test_config, input_data=input_data,
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

