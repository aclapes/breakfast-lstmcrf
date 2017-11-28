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

class SimpleCrfModel(object):
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
                # indices_b = [indices[perm[ptr_b+i]] for i in range(batch_size)]  # debug

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

                yield (np.array(video_features_b), np.array(outputs_b), lengths_b)

        # Reading the data
        # <-- (ONLY DEBUG, not working on full run)
        # g = gen()
        # for i in range(10):
        #     x_batch, y_batch, l_batch = g.next()
        # ---
        with tf.device('/cpu:0'):
            self.iterator = (
                tf.data.Dataset.from_generator(
                    batch_generator,
                    (tf.float32, tf.int32, tf.int32),
                    (
                        tf.TensorShape([None, None, num_features]),
                        tf.TensorShape([None, None]),
                        tf.TensorShape([None]),
                    )
                )
                .prefetch(2)
                .repeat()
            ).make_one_shot_iterator()

            x_batch, y_batch, l_batch = self.iterator.get_next()
        # -->

        # -----------------------------------
        # CRF
        # -----------------------------------

        matricied_x = tf.reshape(x_batch, [-1, num_features])
        proj_w = tf.get_variable('proj_w', [num_features, num_classes], dtype=tf.float32)
        proj_b = tf.get_variable('proj_b', [num_classes], dtype=tf.float32)
        output = tf.matmul(matricied_x, proj_w) + proj_b

        # Decide wheter or not use scores for CRF unary terms
        # <-- (YES)
        # unary_params = tf.get_variable("unary_params", shape=[x_batch.get_shape()[-1]], dtype=tf.float32)
        # unary_scores = tf.multiply(x_batch, unary_params)
        # --- (NO)
        unary_scores = tf.reshape(output, [-1, tf.shape(x_batch)[1], num_classes])
        # -->

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_params = crf.crf_log_likelihood(
            unary_scores, y_batch, l_batch)

        # -----------------------------------
        # Evaluation and training
        # -----------------------------------

        with tf.name_scope('cost'):
            # compute loss and framewise predictions
            self.loss = tf.reduce_mean(-log_likelihood)
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

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # -----------------------------------
        # Tensorboard's auxiliary stuff
        # -----------------------------------
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

                # print(' -> loss=%.5f, acc(mof)=%2.2f%%' % (vals['loss'], 100.0 * vals['acc']))
                self.writer.add_summary(vals['summaries'], global_step=epoch_nr * num_batches + batch_i)
                batch_i += 1
            except (tf.errors.OutOfRangeError, exceptions.StopIteration) as e:
                break
        bar.finish()

        mean_loss = np.mean(batch_loss)
        mean_acc = np.mean(batch_accs)

        return (mean_loss, mean_acc) #, 100*(hit_classes/true_classes)


class SimpleCrfPipeline(object):
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
                with tf.variable_scope('Model', initializer=tf.random_uniform_initializer(-0.01, 0.01)):
                    self.train_model = SimpleCrfModel(config=config, input_data=input_data,
                                                          test_subset=test_subset,
                                                          summaries_dir=path.join(logging_path, 'train'),
                                                          is_training=True)
            with tf.name_scope('Validation'):
                with tf.variable_scope('Model', reuse=True):
                    self.val_model = SimpleCrfModel(config=val_config, input_data=input_data,
                                                        test_subset=test_subset,
                                                        summaries_dir=path.join(logging_path, 'validation'),
                                                        is_training=False)
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True):
                    self.te_model = SimpleCrfModel(config=test_config, input_data=input_data,
                                                       test_subset=test_subset,
                                                       summaries_dir=path.join(logging_path, 'test'),
                                                       is_training=False)

            self.init_op = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()


    def run(self, gpu_options, restore_from_ckpt=None):
        np.set_printoptions(precision=2,linewidth=200)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)) as session:
            if restore_from_ckpt is None:
                restore_from_ckpt = tf.train.latest_checkpoint(self.output_models_path)
            try:
                self.saver.restore(
                    session,
                    restore_from_ckpt
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

                if e+1 in set([10, 50, 100, 150, 200, 250, self.num_epochs]):
                    # Save the model
                    # self.saver.save(
                    #     session,
                    #     path.join(self.output_models_path, 'ckpt'),
                    #     global_step=e,
                    # )
                    # print 'Model saved'

                    loss_te, mof_te = self.te_model.run_epoch(session, e)
                    print('[Testing epoch] loss=%.5f, acc=%2.2f%%' % (loss_te, 100.0 * mof_te))

