import os

import h5py
import numpy as np
import tensorflow as tf
from progressbar import ProgressBar
from tensorflow.contrib import rnn

from src.evaluation import compute_framewise_accuracy, compute_classwise_accuracy
from src.preprocessing import compute_class_weights
from src.reader import read_data_generator

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.set_printoptions(precision=2, linewidth=150)

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
        decay_rate = config['decay_rate']
        hidden_size = config['hidden_size']
        drop_prob = config['drop_prob']
        clip_norm = config['clip_norm']

        self.class_weights = config['class_weights']

        # Graph construction

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch = tf.placeholder(tf.float32, shape=[None, None, num_features])
        self.y_batch = tf.placeholder(tf.int32, shape=[None, None])
        self.l_batch = tf.placeholder(tf.int32, shape=[None])

        # self.state_placeholder = tf.placeholder(tf.float32, shape=[2, 2, batch_size, hidden_size])

        # x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        x_batch = self.x_batch
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

        h_states, self.final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            x_batch,
            dtype=tf.float32,
            swap_memory=True,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            sequence_length=self.l_batch  # do not process padded parts
        )

        self.h_states = tf.concat(h_states, axis=2)

        # rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
        #     cell,
        #     x_batch,
        #     dtype=tf.float32,
        #     initial_state=self.initial_state,  # statefull rnn
        #     sequence_length=self.l_batch  # do not process padded parts
        # )

        matricied_x = tf.reshape(self.h_states, [-1, 2*hidden_size])  # using bidirectional -> 2x hidden_size
        softmax_w = tf.get_variable('softmax_w', [2*hidden_size, no_classes], dtype=tf.float32, regularizer=tf.contrib.layers.l1_regularizer(scale=0.001))
        softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(matricied_x, softmax_w) + softmax_b

        # Class weighting
        # See: https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow#answer-38912982
        # ---
        y_onehot = tf.one_hot(self.y_batch, no_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_onehot_matricied = tf.reshape(y_onehot, [-1, no_classes])

        classweights = tf.expand_dims(tf.constant(self.class_weights, dtype=y_onehot_matricied.dtype), axis=0)
        weight_per_label = tf.transpose(tf.matmul(y_onehot_matricied, tf.transpose(classweights)))
        xent = tf.multiply(weight_per_label,
                      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot_matricied))
        # ---

        # reshape to sequence format and mask padded timesteps
        xent = tf.reshape(xent, [-1,tf.shape(self.y_batch)[1]])
        mask = tf.sequence_mask(self.l_batch)
        masked_xent = tf.boolean_mask(xent, mask)

        # compute loss and framewise predictions
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(masked_xent) + tf.add_n(reg_losses)

        self.logits = tf.reshape(logits, [-1,tf.shape(self.y_batch)[1],no_classes])
        self.predictions = tf.argmax(self.logits, 2)

        if not is_training:
            return

        global_step = tf.Variable(0, trainable=False)
        boundaries = (np.array([100, 1000], dtype=np.int32) * batch_size).tolist()
        values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
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
                                          self.input_data['lengths'][:,-1],
                                          batch_size=self.config['batch_size'])

        num_instances = self.input_data['video_features'].shape[0]
        num_batches = int(np.ceil(num_instances / float(self.config['batch_size'])))
        batch_loss = [None] * num_batches
        batch_accs = [None] * num_batches

        # state = session.run(self.initial_state)
        # state = np.zeros((2, 2, self.config['batch_size'], self.config['hidden_size']), dtype=np.float32)

        fetches = {
            'cost': self.loss,
            'h_states' : self.h_states,
            'logits' : self.logits,
            # 'final_state': self.final_state,
            'predictions': self.predictions,
        }
        if self.is_training:
            fetches['train_op'] = self.train_op
            # fetches['curr_learn_rate'] = self.curr_learn_rate
            # fetches['grads'] = self.grads

        hit_classes = np.zeros((len(self.class_weights),), dtype=np.float32)
        true_classes = np.zeros((len(self.class_weights),), dtype=np.float32)

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            progbar.update(b)
            batch = self.reader.next()

            # c, h = self.initial_state
            # feed_dict[c] = state.c
            # feed_dict[h] = state.h
            feed_dict = {self.x_batch: batch[0], self.y_batch: batch[1], self.l_batch: batch[2]}
                         # self.state_placeholder: state}

            vals = session.run(fetches=fetches, feed_dict=feed_dict)

            # print ' -> ', np.unique(vals['predictions'])
            # print vals['final_state'].h[0,:3]
            # state = vals['final_state']
            # if self.is_training:
            #     print vals['curr_learn_rate']

            batch_loss[b] = vals['cost']
            batch_accs[b] = compute_framewise_accuracy(vals['predictions'], batch[1], batch[2])
            hits, trues = compute_classwise_accuracy(vals['predictions'], batch[1], batch[2], self.class_weights)
            hit_classes += hits
            true_classes += trues
            # if num_batches < num_instances:
            #     print 100.* (hits/trues)

            progbar.update(b)
        progbar.finish()

        return (np.mean(batch_loss), np.mean(batch_accs)), 100 * (hit_classes / true_classes)


    def write_epoch(self, session, output_file):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        f_output = h5py.File(output_file, 'w')

        self.reader = read_data_generator(self.input_data['video_features'],
                                          self.input_data['outputs'],
                                          self.input_data['lengths'][:,-1],
                                          batch_size=self.config['batch_size'])

        batch_size = self.config['batch_size']
        num_instances = self.input_data['video_features'].shape[0]
        num_batches = int(np.ceil(num_instances / float(batch_size)))
        num_words = self.config['num_words']
        hidden_size = self.config['hidden_size']
        num_classes = self.config['no_classes']

        f_output.create_dataset('hidden_states',(num_instances, num_words, 2*hidden_size), fillvalue=0,
                            dtype=np.float32, chunks=(batch_size, num_words//100, 2*hidden_size),
                            compression='gzip', compression_opts=9)

        f_output.create_dataset('logits', (num_instances, num_words, num_classes), fillvalue=0,
                            dtype=np.float32, chunks=(batch_size, num_words//100, num_classes),
                            compression='gzip', compression_opts=9)

        f_output.create_dataset('predictions', (num_instances, num_words), fillvalue=255,
                            dtype=np.uint8, chunks=(batch_size, num_words//100))

        f_output.create_dataset('labels', data=self.input_data['outputs'], dtype=np.int32)
        f_output.create_dataset('lengths', data=self.input_data['lengths'],dtype=np.int32)

        fetches = {
            'h_states' : self.h_states,
            'logits' : self.logits,
            'predictions': self.predictions,
        }

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            progbar.update(b)
            batch = self.reader.next()

            feed_dict = {self.x_batch: batch[0], self.y_batch: batch[1], self.l_batch: batch[2]}

            vals = session.run(fetches=fetches, feed_dict=feed_dict)

            st, end = b*batch_size, min((b+1)*batch_size, num_instances)
            f_output['hidden_states'][st:end] = vals['h_states']
            f_output['logits'][st:end] = vals['logits']
            f_output['predictions'][st:end] = vals['predictions']

            progbar.update(b)
        progbar.finish()

        f_output.close()


class SimpleLstmPipeline(object):
    def __init__(self,
                 train,
                 val,
                 te,
                 class_weights_file,
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

        try:
            class_weights = np.load(class_weights_file)
        except IOError, e:
            class_weights = compute_class_weights(train, config['batch_size'])
            np.save(class_weights_file, class_weights)

        config['class_weights'] = class_weights
        config['no_classes'] = len(class_weights)
        self.sorting = np.argsort(class_weights)  # using class weight criterion

        test_config = config.copy()
        test_config['batch_size'] = 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-0.01, 0.01)

            with tf.name_scope('Train'):
                with tf.variable_scope('Model', reuse=False, initializer=initializer):
                    self.train_model = SimpleLstmModel(config=config, input_data=train, is_training=True)
            with tf.name_scope('Validation'):
                with tf.variable_scope('Model', reuse=True):
                    self.val_model = SimpleLstmModel(config=config, input_data=val, is_training=False)
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True):
                    self.te_model = SimpleLstmModel(config=test_config, input_data=te, is_training=False)

            self.init_op = tf.global_variables_initializer()


    def run(self, gpu_options):
        np.set_printoptions(precision=2,linewidth=150)
        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(self.init_op)

            train_evals = np.zeros((self.num_epochs,3), dtype=np.float32)
            val_evals = np.zeros((self.num_epochs,3), dtype=np.float32)

            mof_val_max = 0.

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e + 1, self.num_epochs))

                # Train step
                (loss_train, mof_train), train_class_evals = self.train_model.run_epoch(session)
                moc_train = np.nanmean(train_class_evals)

                # Validation step
                (loss_val, mof_val), val_class_evals = self.val_model.run_epoch(session)
                moc_val = np.nanmean(val_class_evals)

                # Print summary
                print(
                    'TRAIN (loss/mof/moc): %.4f/%.2f%%/%.2f%%, VAL (loss/mof/moc): %.4f/%.2f%%/%.2f%%' % (
                        loss_train, mof_train, moc_train, loss_val, mof_val, moc_val
                    )
                )

                # Print per-class accuracies
                print train_class_evals[self.sorting]
                print val_class_evals[self.sorting]

                # Keep track of loss/mof/moc across epochs
                train_evals[e,:] = [loss_train, mof_train, moc_train]
                val_evals[e,:] = [loss_val, mof_val, moc_val]

                # Train step (every few epochs). To see progress (not choosing based on this!)
                if e > 20 and mof_val > mof_val_max:
                    (loss_te, mof_te), te_class_evals = self.te_model.run_epoch(session)
                    moc_te = np.nanmean(te_class_evals)
                    print te_class_evals[self.sorting]
                    print('TE (mof/moc): %.2f%%/%.2f%%' % (mof_te,moc_te))

                    print('Writing outputs...')
                    output_dir = os.path.join('/data/datasets/breakfast/lstm_outputs_pooled-uFV_20win_50ovl/', str(e).zfill(6))
                    try:
                        os.makedirs(output_dir)
                    except:
                        pass
                    self.train_model.write_epoch(session, os.path.join(output_dir, 'training_lstm_output.h5'))
                    self.val_model.write_epoch(session, os.path.join(output_dir, 'validation_lstm_output.h5'))
                    self.te_model.write_epoch(session, os.path.join(output_dir, 'testing_lstm_output.h5'))

                    mof_val_max = mof_val

            (_, mof_te), te_class_evals = self.te_model.run_epoch(session)
            moc_te = np.nanmean(te_class_evals)
            print(
                'TRAIN (mof/moc): %.2f%%/%.2f%%, VAL (mof/moc): %.2f%%/%.2f%%, TE (mof/moc): %.2f%%/%.2f%%' % (
                    np.nanmean(train_evals[:,1]), np.nanmean(train_evals[:,2]),
                    np.nanmean(val_evals[:,1]), np.nanmean(val_evals[:,2]),
                    mof_te, moc_te)
            )