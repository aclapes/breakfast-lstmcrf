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

        max_segments = 50

        # smallest positive float32 and float64, used to avoid numerical issues with tf with -inf when log(0)
        tiniest_f32 = np.finfo(np.float32).tiny
        tiniest_f64 = np.finfo(np.float64).tiny

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
                    outputs = np.pad(instance['outputs'], (0, maxlen_b-length), 'constant', constant_values=0)
                    transitions = np.pad(instance['transitions'], (0, maxtrans_b-len(instance['transitions'])), 'constant', constant_values=-1)

                    video_features_b.append(video_features)
                    outputs_b.append(outputs)
                    transitions_b.append(transitions)

                yield (np.array(video_features_b), np.array(outputs_b), np.array(lengths_b), np.array(transitions_b))

        # Reading the data
        # <-- (ONLY DEBUG, not working on full run)
        g = batch_generator()
        for i in range(10):
            x_batch, y_batch, l_batch, t_batch = g.next()
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

        self.x_batch = x_batch
        self.y_batch = y_batch
        self.l_batch = l_batch
        self.t_batch = t_batch
        # -----------------------------------
        # Scrf
        # -----------------------------------

        matricied_x = tf.reshape(x_batch, [-1, num_features])
        proj_w = tf.get_variable('proj_w', [num_features, num_classes-1], dtype=tf.float32)
        proj_b = tf.get_variable('proj_b', [num_classes-1], dtype=tf.float32)
        output = tf.matmul(matricied_x, proj_w) + proj_b
        output = tf.nn.softmax(output)
        self.output = output

        x_batch = tf.reshape(output, [-1, tf.shape(x_batch)[1], num_classes-1])

        x_itgl = tf.cumsum(x_batch, axis=1, reverse=True)
        x_itgl = tf.pad(x_itgl, [[0,0],[0,1],[1,0]], 'CONSTANT', constant_values=0.0)   # create 0 scores for zeroth dummy class
        # zero_col =  tf.fill([tf.shape(x_batch)[0],1,tf.shape(x_batch)[2]], 0.)
        # x_itgl = tf.concat([x_itgl, zero_col], axis=1)

        # with tf.device('/gpu:0'):
        #     self.transitions_placeholder = tf.placeholder(tf.int32, [num_classes, num_classes])
        transitions = tf.constant(input_data['transition_matrix'])

        # Prepare reordering for sequence score computation
        # <--
        x_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(x_itgl)[0])
        p_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(x_itgl)[0])
        end_inds = tf.TensorArray(dtype=tf.int32, size=tf.shape(x_itgl)[0])

        def cond(i, *args): return i < x_ta.size()

        def loop(i, X_, t_, T, X_out_ta, probs_out_ta, end_inds_ta):
            # get segment labels within the batch's sequence
            segm_lbls = tf.boolean_mask(t_[i], t_[i] >= 0)  # eliminate padded values
            num_segms = tf.shape(segm_lbls)[0]
            padding = tf.shape(t_[i])[0] - num_segms  # but keep padding amount for later re-stacking
            # gather segment scores
            segm_scores = tf.gather(X_[i], segm_lbls, axis=1)
            segm_scores_pad = tf.pad(segm_scores, [[0,0],[0,padding]], 'CONSTANT')
            X_out_ta = X_out_ta.write(i, segm_scores_pad)

            # get the transition probabilities between contiguous segments
            trans_from_to_pairs = tf.concat(  #  from segment labels to transitions, e.g. "[0,1,34,3,..]" -> "[[0,1],[1,34],[34,3],[3,..
                [tf.expand_dims(segm_lbls[:-1], axis=1),
                 tf.expand_dims(segm_lbls[1:], axis=1)],
                axis=1
            )

            # trans_from_to_pairs_P = tf.Print(trans_from_to_pairs, [trans_from_to_pairs], 'trans_from_to_pairs: ')

            trans_probs = tf.gather_nd(T, trans_from_to_pairs)  # look into transitions matrix

            trans_probs_pad = tf.pad(trans_probs, [[0,padding]], 'CONSTANT', constant_values=0)  # re-pad
            # trans_probs_pad = tf.concat([[1.], trans_probs_pad], axis=0)
            probs_out_ta = probs_out_ta.write(i, trans_probs_pad)

            end_inds_ta = end_inds_ta.write(i, tf.stack([i, l_batch[i], num_segms-1]))

            return i+1, X_, t_, T, X_out_ta, probs_out_ta, end_inds_ta

        _, _, _, _, batch_ta, probs_ta, end_inds_ta = tf.while_loop(
            cond, loop,
            loop_vars=[0, x_itgl, t_batch, transitions, x_ta, p_ta, end_inds])
        self.x_batch_reord = batch_ta.stack()
        self.p_batch = probs_ta.stack()
        self.end_inds_batch = end_inds_ta.stack()
        # -->
        #
        # # self.x_batch_reord = tf.pad(self.x_batch_reord, [[0,0],[0,1],[0,0]], 'CONSTANT', constant_values=0.0)
        #
        # # alphas0 = tf.fill([tf.shape(x_batch)[0],, 1.0)
        # # idx = tf.Variable(2, trainable=False)
        #
        #
        time_steps = tf.shape(x_batch)[1]
        #
        #
        #
        #
        # # betas_initial = tf.fill([tf.shape(self.x_batch_reord)[0], 1, tf.shape(self.x_batch_reord)[2]], -np.finfo(np.float32).max)
        # # ones = tf.fill([tf.shape(self.x_batch_reord)[0], 1, 1], .0)
        # # zeros = tf.fill([tf.shape(self.x_batch_reord)[0], 1, 1], -np.finfo(np.float32).max)
        # # betas_initial = tf.concat([ones,betas_initial], axis=2)
        # # betas_initial = tf.expand_dims(self.x_batch_reord[:,0,:] - self.x_batch_reord[:,1,:], axis=1)
        #

        J = tf.shape(self.x_batch_reord)[2]

        betas_0 = tf.scatter_nd(
            tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0, 0], [0, 2]], 'CONSTANT', constant_values=0.0),
            tf.ones([batch_size], dtype=tf.float64) - tiniest_f64,
            shape=[batch_size, 1, J],
        )

        betas_0 = tf.log(betas_0 + tiniest_f64)

        def should_continue(t, *args):
            return t <= time_steps

        def iteration(t, betas):
            # alphas_print = tf.Print(alphas_, [t, tf.shape(alphas_)], 'tf.shape(tmp): ')
            # betas_ = tf.expand_dims(betas, 1)

            # scores_ = tf.slice(self.x_batch_reord, [0, 0, 0], [-1, t, -1], name='hello')
            # scores_offset = tf.slice(self.x_batch_reord, [0, t + 1, 0], [-1, 1, -1], name='world')
            scores_ = (self.x_batch_reord[:,:t,1:] - self.x_batch_reord[:,t:(t+1),1:]) / tf.cast(t,tf.float32)


            tmp = betas[:,:,:-1] + tf.cast(tf.expand_dims(self.p_batch, 1) * scores_, dtype=tf.float64)

            betas_t = tf.reduce_logsumexp(tmp, axis=1, keep_dims=True)
            betas_t = tf.pad(betas_t, [[0,0],[0,0],[1,0]], 'CONSTANT', constant_values=np.log(tiniest_f64))

            return t+1, tf.concat([betas, betas_t], axis=1)

        initial_t = tf.constant(1, dtype=tf.int32)
        self.t_final, self.betas_final = tf.while_loop(
            should_continue, iteration,
            loop_vars=[initial_t, betas_0],
            shape_invariants=[initial_t.get_shape(), tf.TensorShape([None, None, None])])

        self.betas_prefinal = self.betas_final
        self.betas_final = (tf.gather_nd(self.betas_final, self.end_inds_batch))
        #
        #
        #
        #
        #
        #
        # # initial_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps+1, clear_after_read=False)
        # # acc = tf.fill([tf.shape(x_batch)[0],tf.shape(x_batch)[2]], 0.)
        # # initial_outputs = initial_outputs.write(0, acc)
        # # initial_t = tf.constant(1, dtype=tf.int32)
        # #
        # # def should_continue(t, *args):
        # #     return t <= time_steps
        # #
        # # def iteration(t, input, outputs_):
        # #     input_at_t = tf.squeeze(tf.slice(input,[0,t,0],[-1,1,-1]), 1)
        # #     acc = outputs_.read(t-1) + input_at_t
        # #     outputs_ = outputs_.write(t, acc)
        # #     return t+1, input, outputs_
        # #
        # # _, _, outputs_ta = tf.while_loop(
        # #     should_continue, iteration,
        # #     [initial_t, x_batch, initial_outputs])
        #
        #
        # # x_itgl_pt = tf.Print(x_itgl, [tf.shape(x_batch),tf.shape(x_itgl)], 'Message: ')
        # # with tf.control_dependencies([x_itgl_pt]):
        # #     self.print_xs = tf.no_op()
        #
        # # initial_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)
        # # initial_outputs = initial_outputs.write(0, x_batch[:,0,:])
        #
        # # alphas = tf.get_variable("alphas", tf.shape(x_batch), dtype=tf.float32, initializer=tf.zeros_initializer, trainable=Fals)
        # # tf.get_variable("alphas2", shape=[None, None, num_classes], dtype=tf.float32, trainable=False,
        # #                 validate_shape=False)
        # # alphas = tf.Variable(tf.zeros(tf.shape(x_batch)), dtype=tf.float32, validate_shape=False, trainable=False)
        # # self.assign_op = alphas[:,0,:].assign(x_batch[:,0,:])
        #
        #
        #
        #
        #
        alphas_0 = tf.scatter_nd(
            tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0, 0], [0, 2]], 'CONSTANT', constant_values=0.0),
            tf.ones([batch_size], dtype=tf.float64) - tiniest_f64,
            shape=[batch_size, 1, num_classes],
        )

        self.alphas_0 = tf.log(alphas_0 + tiniest_f64)

        self.transition_params_ = tf.expand_dims(tf.expand_dims(transitions, axis=0), axis=0)

        def should_continue(t, *args):
            return t <= time_steps

        def iteration(t, alphas):
            alphas_ = tf.expand_dims(alphas,3)
            # alphas_print = tf.Print(alphas_, [t, tf.shape(alphas_)], 'tf.shape(tmp): ')

            # scores_ = tf.slice(x_itgl, [0,1,0], [-1,t,-1]) - tf.slice(x_itgl, [0,t+1,0], [-1,1,-1])
            scores_ = (self.x_batch_reord[:,:t,:] - self.x_batch_reord[:,t:(t + 1),:]) / tf.cast(t, tf.float32)

            # tmp = tf.expand_dims(scores_, axis=-1) * self.transition_params_
            # tmp = alphas_ + tf.cast(self.transition_params_, dtype=tf.float64)
            # tmp2 = tf.cast(scores_, dtype=tf.float64) + tf.reduce_logsumexp(tmp, axis=2)
            tmp = (tf.expand_dims(scores_, 2) * self.transition_params_)  # [B,L,1,C] * [1,1,C,C]
            tmp2 = tmp + tf.cast(tf.expand_dims(alphas, 2), tf.float32) # [B,L,C,C] * [B,L,C,1]

            # s0 * p00 +a0  s1 * p01 + a0
            # s0 * p10 +a1  s1 * p11 + a1
            # s0 * p20 +a2  s1 * p21 + a2

            tmp2 = tf.reduce_logsumexp(tmp2, axis=2) # [B,L,C]

            new_alphas = tf.reduce_logsumexp(tmp2, axis=1, keep_dims=True) # [B,1,C]
            return t+1, tf.concat([alphas,new_alphas], axis=1)

        self.t_final, self.alphas = tf.while_loop(
            should_continue, iteration,
            loop_vars=[initial_t, alphas_0],
            shape_invariants=[initial_t.get_shape(), tf.TensorShape([None, None, num_classes])])
        #
        # # self.alphas_prefinal = tf.slice(self.alphas, [0,tf.shape(self.alphas)[1]-1,0], [-1,1,-1])
        # self.alphas_prefinal = self.alphas[:,-1,:]
        # self.alphas_final = tf.reduce_logsumexp(self.alphas_prefinal, 1)





        # batch_size = tf.shape(x_batch)[0]

        # indices = tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0,0], [0,2]], 'CONSTANT')
        # updates = tf.ones([batch_size], dtype=tf.float32)
        # alphas = tf.scatter_nd(indices, updates, shape=[batch_size,1,num_classes])
        #
        # transition_params_ = (tf.expand_dims(tf.expand_dims(transitions, axis=0), axis=0))
        #
        # def should_continue(t, *args):
        #     return t < time_steps
        #
        # def iteration(t, alphas):
        #     scores_ = tf.slice(x_itgl, [0, 0, 0], [-1, t, -1])
        #     scores_offset = tf.slice(x_itgl, [0, t+1, 0], [-1, 1, -1])
        #     scores_ = (scores_ - scores_offset) /tf.cast(t,tf.float32)
        #
        #     alphas_ = tf.expand_dims(alphas, 3)
        #     tmp = alphas_ + transition_params_
        #     tmp2 = scores_ + tf.reduce_logsumexp(tmp, axis=2)
        #     # tmp2 = alphas + scores_
        #
        #     alphas_t = tf.reduce_logsumexp(tmp2, axis=1, keep_dims=True)
        #
        #     return t + 1, tf.concat([alphas, alphas_t], axis=1)
        #
        # initial_t = tf.constant(1, dtype=tf.int32)
        # self.t_final, self.alphas = tf.while_loop(
        #     should_continue, iteration,
        #     loop_vars=[initial_t, alphas],
        #     shape_invariants=[initial_t.get_shape(), tf.TensorShape([None, None, num_classes])])
        #
        # # alphas_final = tf.slice(self.alphas, [0, tf.shape(self.alphas)[1] - 1, 0], [-1, 1, -1])
        # alphas_final = tf.gather(self.alphas, self.end_inds_batch[:,1], axis=1)
        # alphas_final = tf.boolean_mask(alphas_final, tf.cast(tf.eye(tf.shape(alphas_final)[1]), dtype=tf.bool))
        # self.alphas_final = tf.reduce_logsumexp(alphas_final, axis=1)
        #
        # log_likelihood = self.betas_final - self.alphas_final
        #
        # with tf.name_scope('cost'):
        #     # compute loss and framewise predictions
        #     self.loss = tf.reduce_mean(-log_likelihood)
        #     tf.summary.scalar('loss', self.loss)
        #
        # # with tf.name_scope('evaluation'):
        # #     # self.predictions, _ = Scrf.Scrf_decode(unary_scores, transition_params, l_batch)
        # #     self.predictions = None # Fix
        # #     self.y_batch = y_batch
        # #     self.l_batch = l_batch
        # #     equality = tf.cast(tf.equal(self.predictions, y_batch), tf.float32)
        # #     mask = tf.sequence_mask(l_batch, tf.shape(y_batch)[1])
        # #     self.acc = tf.reduce_sum(tf.boolean_mask(equality, mask)) / tf.reduce_sum(tf.cast(mask, tf.float32))
        # #     tf.summary.scalar('accuracy', self.acc)
        # #
        # self.global_step = tf.Variable(0, trainable=False)
        # boundaries = (np.array([5,10], dtype=np.int32) * batch_size).tolist()
        # values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        # curr_learn_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, name=None)
        #
        # if optimizer_type == 'sgd':
        #     self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=curr_learn_rate)
        # elif optimizer_type == 'adam':
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)
        #
        # # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        #
        # tvars = tf.trainable_variables()
        # self.grads = tf.gradients(self.loss, tvars)
        # grads_clipped, _ = tf.clip_by_global_norm(self.grads, clip_norm=1.0)
        # self.train_op = self.optimizer.apply_gradients(zip(grads_clipped, tvars), global_step=self.global_step)
        # #
        # # # -----------------------------------
        # # # Tensorboard's auxiliary stuff
        # # # -----------------------------------
        # # self.merge_summaries_op = tf.summary.merge_all()
        # # self.writer = tf.summary.FileWriter(summaries_dir, tf.get_default_graph())


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
                v = session.run([self.alphas_0, self.transition_params_])
                # print(' -> loss=%.5f' % (v[-1]))
                # batch_loss.append(v[-1])
                # if np.isnan(v[-1]):
                #     error = 1
                # else:
                #     error = 0
                # self.writer.add_summary(vals['summaries'], global_step=epoch_nr * num_batches + batch_i)
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
        config['num_classes'] = len(class_weights)

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

