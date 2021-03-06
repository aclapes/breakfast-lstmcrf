import argparse
import os

import h5py
import numpy as np
import tensorflow as tf
from progressbar import ProgressBar

from src import c3d_model
from src.evaluation import compute_accuracy, foo
from src.preprocessing import compute_class_weights, compute_mean_channels
from src.reader import read_image_generator

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.set_printoptions(precision=2, linewidth=150)
MOVING_AVERAGE_DECAY = 0.9999

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels, class_weights=None):

  if class_weights is None:
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
  else:
    no_classes = len(class_weights)
    labels_onehot = tf.one_hot(labels, no_classes, on_value=1.0, off_value=0.0, axis=-1)
    classweights = tf.expand_dims(tf.constant(class_weights, dtype=tf.float32), axis=0)
    weight_per_label = tf.transpose(tf.matmul(labels_onehot, tf.transpose(classweights)))
    xent = tf.multiply(weight_per_label,
                       tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_onehot))

  cross_entropy_mean = tf.reduce_mean(xent)
  tf.summary.scalar(
                  name_scope + 'cross entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
  tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)
  tf.add_to_collection('losses', cross_entropy_mean)
  losses = tf.get_collection('losses', name_scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  tf.summary.scalar(name_scope + 'total loss', total_loss)

  # Compute the moving average of all individual losses and the total loss.
  # loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
  # loss_averages_op = loss_averages.apply(losses + [total_loss])
  # with tf.control_dependencies([loss_averages_op]):
  #   total_loss = tf.identity(total_loss)
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), tf.cast(labels,dtype=tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer, trainable=True):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, trainable=True):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(), trainable=trainable)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


class SimpleCnnModel(object):
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
        channel_means = config['channel_means']

        self.class_weights = config['class_weights']

        # Graph construction
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # Features, output labels, and binary mask of valid timesteps
        self.x_batch_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS],
                                                  name='x_batch_placeholder')
        self.y_batch_placeholder = tf.placeholder(tf.int32,
                                                  shape=[None, c3d_model.NUM_FRAMES_PER_CLIP],
                                                  name='y_batch_placeholder')


        # self.channels_sum = tf.reduce_sum(tf.cast(x_batch_rs, tf.float32), axis=[0,1,2])
        self.channel_means_op = tf.reduce_mean(tf.cast(self.x_batch_placeholder, tf.float32), axis=[0, 1, 2, 3])
        # vgg = tf.contrib.slim.nets.vgg
        # VGG_MEAN = [123.68, 116.78, 103.94]
        # with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=1e-5)):
        #     logits, _ = vgg.vgg_16(self.x_batch, num_classes=no_classes, is_training=is_training,
        #                            dropout_keep_prob=config['drop_prob'])
        #
        # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        # self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn('vgg_16.ckpt', variables_to_restore)
        #
        # fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        # self.fc8_init = tf.variables_initializer(fc8_variables)
        #
        #
        # self.loss =  tf.losses.sparse_softmax_cross_entropy(labels=self.y_batch, logits=logits)
        #
        # fc8_optimizer = tf.train.GradientDescentOptimizer(args.learn_rate)
        # self.train_op = fc8_optimizer.minimize(self.loss, var_list=fc8_variables)

        # y_cnn_labels = tf.reshape(self.y_batch, [-1, 16])

        x_normalized = self.x_batch_placeholder - channel_means

        # y_chunked = tf.reshape(y_batch_split[gpu_index], [-1, 16])
        y_sum = tf.reduce_sum(tf.one_hot(self.y_batch_placeholder, depth=no_classes, dtype=tf.int32), axis=1)
        self.y = tf.argmax(y_sum, axis=1)

        with tf.control_dependencies([self.y]):
            self.no_op = tf.no_op()

        gpu_num = 1
        # x_batch_split = tf.split(self.x_batch, num_or_size_splits=gpu_num, axis=0)
        # y_batch_split = tf.split(self.y_batch, num_or_size_splits=gpu_num, axis=0)

        tower_grads1 = []
        tower_grads2 = []
        logits = []
        losses = []
        opt1 = tf.train.AdamOptimizer(1e-4)
        opt2 = tf.train.AdamOptimizer(2e-4)
        # for gpu_index in range(0, gpu_num):
            # with tf.device('/gpu:%d' % gpu_index):
        with tf.name_scope('%s_%d' % ('dextro-research', 0)) as scope: #gpu_index)) as scope:
    #         with tf.variable_scope('var_name', reuse=(None if gpu_index == 0 else True)):
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005, trainable=False),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005, trainable=False),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005, trainable=False),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005, trainable=False),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005, trainable=False),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005, trainable=False),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005, trainable=False),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005, trainable=False),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': _variable_with_weight_decay('wout', [4096, no_classes], 0.0005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000, trainable=False),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000, trainable=False),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000, trainable=False),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000, trainable=False),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000, trainable=False),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000, trainable=False),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000, trainable=False),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000, trainable=False),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                'out': _variable_with_weight_decay('bout', [no_classes], 0.000),
            }

            # x = tf.reshape(x_batch_split[gpu_index], [-1, 16, 224, 224, 3])

            varlist1 = weights.values()
            varlist2 = biases.values()
            logit = c3d_model.inference_c3d(
                x_normalized,
                0.5,
                tf.shape(x_normalized)[0],
                weights,
                biases
            )

            loss = tower_loss(
                scope,
                logit,
                self.y,
                class_weights=config['class_weights']
            )

            grads1 = opt1.compute_gradients(loss, varlist1)
            grads2 = opt2.compute_gradients(loss, varlist2)
            tower_grads1.append(grads1)
            tower_grads2.append(grads2)
            logits.append(logit)
            losses.append(loss)
                        # tf.get_variable_scope().reuse_variables()
        concat_logits_op = tf.concat(logits, 0)
        self.loss = tf.reduce_mean(losses)
        # self.noop = tf.no_op()
        self.tower_accuracy = tower_acc(concat_logits_op, self.y)
        self.preds = tf.argmax(logit, 1)
        tf.summary.scalar('accuracy', self.tower_accuracy)
        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt1.apply_gradients(grads1)
        apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        self.train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)

        saved_variables = []
        variables = weights.values() + biases.values()
        for v in variables:
            var_name = v.name.split(':')[0]
            if var_name != 'Model/wout' and var_name != 'Model/bout':
                saved_variables.append(v)

        self.saver = tf.train.Saver(saved_variables)

        # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['Model/wout', 'Model/bout'])
        # self.init_rest = tf.contrib.framework.assign_from_checkpoint_fn('my-new-model', variables_to_restore)

        # x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
        # if is_training:
        #     x_batch = tf.nn.dropout(x_batch, keep_prob=0.5)  # TODO: experiment with this dropout
        #
        # # Compute unary scores from a linear layer.
        # matricied_x = tf.reshape(x_batch, [-1, num_features])
        #
        # # hidden_w = tf.get_variable('hidden_w', [num_features, config['hidden_size']], dtype=tf.float32)
        # # hidden_b = tf.get_variable('hidden_b', [config['hidden_size']], dtype=tf.float32, initializer=tf.zeros_initializer())
        # # hidden_activations = tf.matmul(matricied_x, hidden_w) + hidden_b
        # # if is_training:
        # #     hidden_activations = tf.nn.dropout(hidden_activations, keep_prob=(1-config['drop_prob']))
        # #
        # # softmax_w = tf.get_variable('softmax_w', [config['hidden_size'], no_classes], dtype=tf.float32)
        # # softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        # # logits = tf.matmul(hidden_activations, softmax_w) + softmax_b
        #
        # softmax_w = tf.get_variable('softmax_w', shape=[num_features, no_classes], dtype=tf.float32, regularizer=tf.contrib.layers.l1_regularizer(scale=0.001))
        # softmax_b = tf.get_variable('softmax_b', [no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
        # logits = tf.matmul(matricied_x, softmax_w) + softmax_b
        #
        # logits = tf.nn.softmax(logits)
        # logits = tf.multiply(logits, self.class_weights)
        #
        # # normalized_logits = tf.nn.softmax(logits)
        # unary_scores = tf.reshape(
        #     logits, [-1, num_words, no_classes]
        # )
        #
        # y_flattened = tf.reshape(self.y_batch, [-1])
        # weights = tf.gather(self.class_weights, y_flattened)
        # weights = tf.reshape(weights, shape=[-1, num_words])
        #
        # # Compute the log-likelihood of the gold sequences and keep the transition
        # # params for inference at test time.
        # log_likelihood, transition_params = crf2.crf_log_likelihood(
        #     unary_scores, self.y_batch, self.l_batch)
        # # Add a training op to tune the parameters.
        #
        # self.decoding, _ = crf2.crf_decode(unary_scores, transition_params, self.l_batch)
        #
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.reduce_sum(-log_likelihood) + tf.add_n(reg_losses)
        # if not is_training:
        #     return
        #
        # global_step = tf.Variable(0, trainable=False)
        # boundaries = (np.array([100, 1000], dtype=np.int32) * batch_size).tolist()
        # values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        # curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)
        #
        # if optimizer_type == 'sgd':
        #     self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=curr_learn_rate)
        # elif optimizer_type == 'adam':
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)
        # elif optimizer_type == 'rmsprop':
        #     self.optimizer = tf.train.RMSPropOptimizer(learning_rate=curr_learn_rate)
        #
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=clip_norm)
        # self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)


    def run_epoch(self, session):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        self.reader = read_image_generator(self.input_data['video_features'],
                                          self.input_data['outputs'],
                                          batch_size=self.config['batch_size'])

        num_instances = self.input_data['video_features'].shape[0]
        num_batches = int(np.ceil(num_instances / float(self.config['batch_size'])))
        batch_loss = [None] * num_batches
        batch_accs = [None] * num_batches

        hit_classes = np.zeros((len(self.class_weights),), dtype=np.float32)
        true_classes = np.zeros((len(self.class_weights),), dtype=np.float32)

        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = self.reader.next()

            fetches = {
                'loss': self.loss,
                'acc' : self.tower_accuracy,
                'predictions' : self.preds,
                'trues' : self.y
            }
            if self.is_training:
                fetches['train_op'] = self.train_op

            vals = session.run(
                fetches,
                feed_dict={self.x_batch_placeholder: batch[0], self.y_batch_placeholder: batch[1]},
            )

            # vals = session.run(
            #     [self.no_op], #[self.train_op, self.tower_accuracy], # [self.x_batch, self.y_batch],
            #     feed_dict={self.x_batch: batch[0], self.y_batch: batch[1]},
            # )

            # print vals['loss']
            batch_loss[b] = vals['loss']
            acc = compute_accuracy(vals['predictions'], vals['trues'])
            print ' -> ' + str(vals['loss']) + ', ' + str(acc)
            batch_accs[b] = acc
            hits, trues = foo(vals['predictions'], vals['trues'], self.class_weights)
            hit_classes += hits
            true_classes += trues
            # if num_batches < num_instances:
            #     print 100.* (hits/trues)

            progbar.update(b)
        progbar.finish()

        return (np.mean(batch_loss), np.mean(batch_accs)), 100*(hit_classes/true_classes)


class SimpleCnnPipeline(object):
    def __init__(self,
                 train,
                 val,
                 te,
                 channel_means_file,
                 no_classes,
                 class_weights_file,
                 batch_size,
                 learn_rate,
                 decay_rate,
                 num_epochs,
                 drop_prob,
                 optimizer_type='adam',
                 clip_norm=1.0):

        self.num_epochs = num_epochs
        self.channel_means_file = channel_means_file

        config = dict(
            no_classes = no_classes,
            batch_size = batch_size,
            num_words = train['video_features'].shape[1],
            num_features = train['video_features'].shape[2:],
            optimizer_type = optimizer_type,
            drop_prob = drop_prob,
            decay_rate = decay_rate,
            learn_rate = learn_rate,
            clip_norm = clip_norm,
            channel_means_file = channel_means_file
        )

        try:
            class_weights = np.load(class_weights_file)
        except IOError, e:
            class_weights = compute_class_weights(train, config['batch_size'])
            np.save(class_weights_file, class_weights)

        config['class_weights'] = class_weights
        self.sorting = np.argsort(class_weights)  # using class weight criterion

        try:
            channel_means = np.load(self.channel_means_file)
        except IOError, e:
            channel_means = compute_mean_channels(train, config['batch_size'])
            np.save(self.channel_means_file, channel_means)

        config['channel_means'] = channel_means

        test_config = config.copy()
        test_config['batch_size'] = c3d_model.NUM_FRAMES_PER_CLIP

        self.graph = tf.Graph()
        with self.graph.as_default():
            initializer = tf.random_uniform_initializer(-0.01, 0.01)
            with tf.variable_scope('Model', reuse=False, initializer=initializer):
                self.train_model = SimpleCnnModel(config=config, input_data=train, is_training=True)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.val_model = SimpleCnnModel(config=config, input_data=val, is_training=False)

            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                self.te_model = SimpleCnnModel(config=test_config, input_data=te, is_training=False)

            self.init_op = tf.global_variables_initializer()

    def run(self, gpu_options):
        # with tf.Graph().as_default(), tf.Session().as_default() as session:
            # ckpt_reader = tf.train.NewCheckpointReader(
            #     '/data/datasets/UCF-101/sports1m_finetuning_ucf101.model')  # .restore(session, '/data/datasets/UCF-101/sports1m_finetuning_ucf101.model')
            #
            # new_vars = []
            # for name in ckpt_reader.get_variable_to_shape_map():
            #     if name != 'var_name/wout' and name != 'var_name/wout':
            #         new_vars.append(tf.Variable(ckpt_reader.get_tensor(name), name=name.replace('var_name', 'Model')))
            #
            # saver = tf.train.Saver(new_vars)
            # session.run(tf.global_variables_initializer())
            # saver.save(session, 'my-new-model')

        with tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(self.init_op)
            if True:
                # ckpt_reader = tf.train.NewCheckpointReader('my-new-model')
                self.train_model.saver.restore(session, 'my-new-model')
                # session.run(self.train_model.init_rest)

            train_evals = np.zeros((self.num_epochs,3), dtype=np.float32)
            val_evals = np.zeros((self.num_epochs,3), dtype=np.float32)

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
                if e in [5, 10, 50, 100, 500, 1000, 2000, 10000, 20000]:
                    (loss_te, mof_te), te_class_evals = self.te_model.run_epoch(session)
                    moc_te = np.nanmean(te_class_evals)
                    print te_class_evals[self.sorting]
                    print('TE (mof/moc): %.2f%%/%.2f%%' % (mof_te,moc_te))

            (_, mof_te), te_class_evals = self.te_model.run_epoch(session)
            moc_te = np.nanmean(te_class_evals)
            print(
                'TRAIN (mof/moc): %.2f%%/%.2f%%, VAL (mof/moc): %.2f%%/%.2f%%, TE (mof/moc): %.2f%%/%.2f%%' % (
                    np.nanmean(train_evals[:,1]), np.nanmean(train_evals[:,2]),
                    np.nanmean(val_evals[:,1]), np.nanmean(val_evals[:,2]),
                    mof_te, moc_te)
            )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform labelling of sequences using a LSTMCRF model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='/data/datasets/breakfast/dataset/',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-m',
        '--channel-means-file',
        type=str,
        dest='channel_means_file',
        default='./breakfast/channel_means.npy',
        help=
        'File (npy) containing a 3-valued vector (default: %(default)s)')

    parser.add_argument(
        '-w',
        '--class-weights-file',
        type=str,
        dest='class_weights_file',
        default='./breakfast/class_weights.npy',
        help=
        'File (npy) containing a N-sized vector, where N number of classes (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=1,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=1e-4,
        help=
        'Starting learning rate. It decays after 100 and 1000 epochs by a factor specified by --decay-rate argument (default: %(default)s)')

    parser.add_argument(
        '-dr',
        '--decay-rate',
        type=float,
        dest='decay_rate',
        default=10,
        help=
        'Decay rate for inverse time decay (default: %(default)s)')

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=2000,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-ot',
        '--optimizer-type',
        type=str,
        dest='optimizer_type',
        default='adam',
        help=
        'Optimizer type (sgd or adam) (default: %(default)s)')

    parser.add_argument(
        '-c',
        '--clip-norm',
        type=float,
        dest='clip_norm',
        default=5.0,
        help=
        'Clipping gradients by norm above clip_norm (default: %(default)s)')

    parser.add_argument(
        '-p',
        '--drop-prob',
        type=float,
        dest='drop_prob',
        default=0.1,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------
    parser.add_argument(
        '-G',
        '--gpu-memory',
        type=float,
        dest='gpu_memory',
        default=0.95,
        help=
        'GPU memory to reserve (default: %(default)s)')

    args = parser.parse_args()
    print args

    # Read breakfast from hdf5 file
    f_training = h5py.File(os.path.join(args.input_dir, 'training.h5'), 'r')
    f_validation = h5py.File(os.path.join(args.input_dir, 'validation.h5'), 'r')
    f_testing = h5py.File(os.path.join(args.input_dir, 'testing.h5'), 'r')

    # st_time = time.time()
    # if not os.path.exists(os.path.join(args.input_dir, 'mean_channels.npy')):
    #     total_sum = np.zeros((f_training['video_features'].shape[-1]), dtype=np.float64)
    #     progress
    #     for i in range(f_training['video_features'].shape[0]):
    #         im = f_training['video_features'][i]
    #         total_sum += np.mean(np.mean(im, axis=1), axis=0)
    # print time.time() - st_time

    # Create a model (choosen via argument passing)
    m = SimpleCnnPipeline(
        f_training,
        f_validation,
        f_testing,
        args.channel_means_file,
        len(f_training['class_weights']),
        args.class_weights_file,
        batch_size=args.batch_size,
        learn_rate=args.learn_rate,
        decay_rate=args.decay_rate,
        num_epochs=args.num_epochs,
        drop_prob=args.drop_prob,
        optimizer_type=args.optimizer_type,
        clip_norm=args.clip_norm
    )

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    m.run(gpu_options)
    # -----------------------------------------------

    f_testing.close()
    f_validation.close()
    f_training.close()
