import argparse
import exceptions
import os

import h5py
import numpy as np
import progressbar
import tensorflow as tf

from src.alexnet import AlexNet
from src.preprocessing import compute_class_weights

np.set_printoptions(precision=2, linewidth=150)
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


def preprocessing(images, labels, batch_size):
    """
    Preprocessing batch function called during tf.data.Dataset.from_generator().
    :param images:
    :param labels:
    :return:
    """

    # Random crops (glimpses)
    # ---
    height = width = 227

    # compute offsets
    # batch_size = tf.shape(images)[0]
    x = tf.random_normal([3 * batch_size])
    y = tf.random_normal([3 * batch_size])
    xx = tf.boolean_mask(x, tf.logical_and(tf.greater_equal(x, -1), tf.less_equal(x, 1)))
    yy = tf.boolean_mask(y, tf.logical_and(tf.greater_equal(y, -1), tf.less_equal(y, 1)))
    offsets = tf.concat([tf.expand_dims(tf.slice(xx, [0], [batch_size]), axis=-1),
                         tf.expand_dims(tf.slice(yy, [0], [batch_size]), axis=-1)],
                        axis=1)

    images = tf.image.extract_glimpse(
        images,
        [height, width],
        offsets,
        uniform_noise=True
    )

    # Left-right clipping of crops
    # ---
    images_stack = tf.unstack(images, axis=0)
    images = tf.stack([tf.image.random_flip_left_right(image) for image in images_stack], axis=0)

    return images, labels


class Simple2DCnnModel(object):
    def __init__(self, config, input_data, is_training, pretrain_weights=None):
        self.config = config
        self.input_data = input_data
        self.is_training = is_training
        self.pretrain_weights = pretrain_weights

        skip_layers = ['fc8']
        train_layers = ['fc7','fc8']

        #
        # # Graph construction
        #

        def gen():
            """ A simple data iterator """
            n =  self.input_data.shape[0]
            perm = np.random.permutation(n)
            end = self.config['batch_size']*(n//self.config['batch_size']) # discard last batch
            perm = perm[:end]
            for idx in perm:
                yield (self.input_data[idx]['image'].astype(np.float32),  self.input_data[idx]['label'])
            return

        with tf.device('/cpu:0'):
            ds = (
                tf.data.Dataset.from_generator(gen,
                                               (tf.float32, tf.int32),
                                               (tf.TensorShape([None,None,3]), tf.TensorShape([])))
                .batch(self.config['batch_size'])
                .map(lambda images, labels: preprocessing(images, labels, self.config['batch_size']))
                .prefetch(4)
            )

            self.iterator = ds.make_initializable_iterator()
            x_batch, y_batch = self.iterator.get_next()


        # # Features, output labels, and binary mask of valid timesteps
        # self.x_batch = tf.placeholder(tf.float32, shape=[None] + list(frame_dims))
        # self.y_batch = tf.placeholder(tf.int32, shape=[None])

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.dropout = tf.placeholder(tf.float32, shape=[])

        # # Using tftables
        # with tf.name_scope('input'):
        #     with tf.device("/cpu:0"):
        #         array_batch_placeholder = self.reader.get_batch(
        #             path = '/dataset',
        #             cyclic = False,
        #             ordered = False
        #         )
        #         x_batch_cpu = tf.to_float(array_batch_placeholder['image'])
        #         y_batch_cpu = array_batch_placeholder['label']
        #
        #         self.loader = self.reader.get_fifoloader(queue_size=1, inputs=[x_batch_cpu, y_batch_cpu], threads=1)
        #
        # x_batch, y_batch = self.loader.dequeue()

        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # boundaries = (np.array([10, 100, 1000], dtype=np.float32) * batch_size).tolist()
        # values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        # curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        if self.config['optimizer_type'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.config['optimizer_type'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise NotImplementedError('Either specify "sgd" or "adam" optimizer type.')

        # input to cnn
        # vgg = vgg19.Vgg19(self.pretrain_weights, fc8_dict=dict(size=num_classes, name='fc8'))
        # vgg.build(x_batch[g*m:(g+1)*m] / 255.0, train_mode=tf.constant(self.is_training, dtype=tf.bool))
        self.alexnet = AlexNet(
            x_batch,
            1.-self.dropout,
            self.config['num_classes'],
            skip_layers,
            train_layers,
            weights_path=self.pretrain_weights
        )

        # output from cnn
        logits = self.alexnet.fc8

        with tf.name_scope("cross_entropy"):
            # Class weighting
            y_onehot = tf.one_hot(y_batch, self.config['num_classes'], on_value=1.0, off_value=0.0, axis=-1)
            classweights = tf.expand_dims(tf.constant(config['class_weights'], dtype=y_onehot.dtype), axis=0)
            weight_per_label = tf.transpose(tf.matmul(y_onehot, tf.transpose(classweights)))
            xent = tf.multiply(weight_per_label,
                               tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot))
            # xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot)

            self.loss = tf.reduce_mean(xent)

        with tf.name_scope('training'):
            self.train_conv5_op = self.get_training_subgraph(['conv5', 'fc6', 'fc7', 'fc8'])
            self.train_fc_all_op = self.get_training_subgraph(['fc6','fc7','fc8'])
            self.train_fc_last_two_op = self.get_training_subgraph(['fc7','fc8'])
            self.train_fc_last_op = self.get_training_subgraph(['fc8'])

        with tf.name_scope("accuracy"):
            preds = (tf.argmax(logits, 1, output_type=tf.int32))
            equality = tf.equal(preds, y_batch)

            self.acc = tf.reduce_mean(tf.cast(equality, tf.float32))

    def get_training_subgraph(self, train_layers):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]
        gradients = tf.gradients(self.loss, var_list)
        return self.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, var_list)),
                                              global_step=self.global_step)

    def reset_global_step(self, session):
        session.run(tf.assign(self.global_step, 0))

    def load_initial_weights(self, session):
        self.alexnet.load_initial_weights(session)

    def get_var_list(self, session):
        return session.run(self.var_names)

    def run_epoch(self, session, tvar_names=None):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        fetches = {
            'loss': self.loss,
            'acc': self.acc,
        }

        feed_dict = {
            self.learning_rate: 1e-5,  # 10x smaller than original learning rate (1e-4)
            self.dropout: (self.config['drop_prob'] if self.is_training else 0)
        }

        if self.is_training:
            if tvar_names == 'TRAIN_FC_ALL':
                fetches['train_op'] = self.train_fc_all_op
            elif tvar_names == 'TRAIN_FC_LAST_TWO':
                fetches['train_op'] = self.train_fc_last_two_op
            elif tvar_names == 'TRAIN_FC_LAST':
                fetches['train_op'] = self.train_fc_last_op
            else:
                fetches['train_op'] = self.train_conv5_op

        batch_loss = []
        batch_accs = []

        num_batches = int(np.ceil(self.config['num_instances'] // self.config['batch_size']))
        session.run(self.iterator.initializer)
        try:
            bar = progressbar.ProgressBar(max_value=num_batches)
            for batch_i in range(num_batches):
                bar.update(batch_i)

                vals = session.run(
                    fetches, feed_dict=feed_dict
                )

                print(' -> loss=%.5f, acc=%2.2f%%' % (vals['loss'], 100.0 * vals['acc']))

                batch_loss.append(vals['loss'])
                batch_accs.append(vals['acc'])
                batch_i += 1
            bar.finish()
        except tf.errors.OutOfRangeError, e:
            raise e
        except exceptions.StopIteration, e:
            raise e

        mean_loss = np.mean(batch_loss)
        mean_acc = np.mean(batch_accs)
        print('[Total epoch] loss=%.5f, acc=%2.2f%%' % (mean_loss, 100.0*mean_acc))

        return (mean_loss, mean_acc) #, 100*(hit_classes/true_classes)


class Simple2DCnnPipeline(object):
    def __init__(self,
                 train_file,
                 val_file,
                 te_file,
                 pretrain_weights_file,
                 batch_size,
                 learn_rate,
                 decay_rate,
                 num_epochs,
                 drop_prob,
                 optimizer_type='adam',
                 clip_norm=1.0,
                 gpu_memory=None):

        train = h5py.File(train_file, 'r')
        val = h5py.File(val_file, 'r')
        test = h5py.File(te_file, 'r')

        self.num_epochs = num_epochs
        self.gpu_memory = gpu_memory

        train_config = dict(
            num_instances = train['/dataset'].shape[0],
            batch_size = batch_size,
            frame_dims = [224,224,3],
            optimizer_type = optimizer_type,
            drop_prob = drop_prob,
            decay_rate = decay_rate,
            learn_rate = learn_rate,
            clip_norm = clip_norm
        )

        # Alternative:
        # <---
        class_weights = train['class_weights'].value
        # ---
        # class weights to alleviate class imbalance during classification
        # try:
        #     class_weights = np.load(class_weights_file)
        # except IOError, e:
        #     class_weights = compute_class_weights(train, train_config['batch_size'])
        #     np.save(class_weights_file, class_weights)
        # --->

        train_config['class_weights'] = class_weights
        train_config['num_classes'] = len(class_weights)
        self.sorting = np.argsort(class_weights)  # for debugging purpouses

        # create config for validation/test models

        val_config = train_config.copy()
        val_config['drop_prob'] = 0.
        val_config['num_instances'] = val['/dataset'].shape[0]

        test_config = train_config.copy()
        test_config['drop_prob'] = 0.
        test_config['num_instances'] = test['/dataset'].shape[0]
        test_config['batch_size'] = 1

        # load pre-trained weights and remove output layer (which has de #classes from ImageNet)
        # pretrain_weights = np.load('/data/datasets/vgg19.npy', encoding='latin1').item()
        # if 'fc8' in pretrain_weights:
        #     del pretrain_weights['fc8']

        self.graph = tf.Graph()
        with self.graph.as_default():
            # initializer = tf.random_uniform_initializer(-0.01, 0.01)
            with tf.variable_scope('Model', reuse=False): #, initializer=initializer):
                self.train_model = Simple2DCnnModel(config=train_config,
                                                    input_data=train['/dataset'],
                                                    is_training=True,
                                                    pretrain_weights=pretrain_weights_file)

            with tf.variable_scope('Model', reuse=True): #, initializer=initializer):
                self.val_model = Simple2DCnnModel(config=val_config,
                                                  input_data=val['/dataset'],
                                                  is_training=False)

            # with tf.variable_scope('Model', reuse=True): #, initializer=initializer):
            #     self.te_model = Simple2DCnnModel(config=test_config, input_data=te, is_training=False)

            # Initialize uninitialized variables (those restored by the saver are not initialized)
            self.init_op = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver()


    def _run(self, session, num_epochs, tvar_names, pretrain=False, save_checkpoints=False, checkpoint_every_n=2):
        """
        ...
        :param session:
        :param num_epochs:
        :param tvar_names:
        :return:
        """

        if pretrain:
            with tf.variable_scope('Model'):
                self.train_model.load_initial_weights(session)

        self.train_model.reset_global_step(session)

        for epoch_i in range(num_epochs):
            print('EPOCH %d/%d' % (epoch_i+1, self.num_epochs))
            # Train
            loss_train, mof_train = self.train_model.run_epoch(session, tvar_names)
            # Validation step
            loss_val, mof_val = self.val_model.run_epoch(session)
            if save_checkpoints and epoch_i % checkpoint_every_n == 0:
                self.saver.save(session, '/data/datasets/breakfast/models/cnn-alexnet/cnn_alexnet-{}'.format(tvar_names),
                                global_step=self.train_model.global_step, write_meta_graph=False)

    def run(self, num_epochs, tvar_names=None):
        """
        ...
        :param num_epochs:
        :param tvar_names:
        :return:
        """
        config_proto = tf.ConfigProto()
        if self.gpu_memory is not None:
            config_proto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory))

        # run a session in a python's context manager ("with" clause)
        with tf.Session(graph=self.graph, config=config_proto) as session:
            try:
                # do we already have a model checkpoint?
                self.saver.restore(session, tf.train.latest_checkpoint('/data/datasets/breakfast/models/cnn-alexnet/'))
                pretrain = False  # if so, do not start on imagenet pretrained weights
            except:
                pretrain = True
                session.run(self.init_op)

            # if tvar_names == 'TRAIN_FC_CASCADE':
                # self._run(session, int((1/3.)*num_epochs), 'TRAIN_FC_LAST', pretrain=pretrain, save_checkpoints=True, checkpoint_every_n=5)
                # self._run(session, int((2/3.)*num_epochs), 'TRAIN_FC_LAST_TWO', pretrain=False, save_checkpoints=True, checkpoint_every_n=5)
            # else:
            self._run(session, num_epochs, tvar_names, pretrain=pretrain, save_checkpoints=True, checkpoint_every_n=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform image classification using a 2D CNN (VGG) model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='/data/datasets/breakfast/images/',
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
        '--pretrain-weights-file',
        type=str,
        dest='pretrain_weights_file',
        default='/data/datasets/bvlc_alexnet.npy',
        help=
        'File (npy) containing pretrain weights for AlexNet (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=64,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=0.01,
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
        default=20,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-v',
        '--trainable-vars',
        type=str,
        dest='tvar_names' ,
        default='',
        help=
        '"TRAIN_FC_LAST", "TRAIN_FC_ALL", "TRAIN_FC_CASCADE" or "" to train all layers (default: %(default)s)')

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
        default=0.5,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------
    parser.add_argument(
        '-G',
        '--gpu-memory',
        type=float,
        dest='gpu_memory',
        default=0.75,
        help=
        'GPU memory to reserve (default: %(default)s)')

    parser.add_argument(
        '-D',
        '--cuda-devices',
        type=str,
        dest='cuda_devices',
        default="3",
        help=
        'GPU devices (default: %(default)s)')

    args = parser.parse_args()
    print args

    # Read breakfast from hdf5 file
    # f_training = h5py.File(os.path.join(args.input_dir, 'training.h5'), 'r')
    # f_validation = h5py.File(os.path.join(args.input_dir, 'validation.h5'), 'r')
    # f_testing = h5py.File(os.path.join(args.input_dir, 'testing.h5'), 'r')

    # reader_te_data = tftables.open_file(filename=os.path.join(args.input_dir, 'testing.h5'), batch_size=args.batch_size)

    if args.cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # Create a model (choosen via argument passing)
    m = Simple2DCnnPipeline(
        os.path.join(args.input_dir, 'training.test.h5'),
        os.path.join(args.input_dir, 'testing.test.h5'),
        os.path.join(args.input_dir, 'testing.test.h5'),
        args.pretrain_weights_file,
        batch_size=args.batch_size,
        learn_rate=args.learn_rate,
        decay_rate=args.decay_rate,
        num_epochs=args.num_epochs,
        drop_prob=args.drop_prob,
        optimizer_type=args.optimizer_type,
        clip_norm=args.clip_norm,
        gpu_memory=args.gpu_memory
    )

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    m.run(args.num_epochs, args.tvar_names)
    # -----------------------------------------------

    # f_testing.close()
    # f_validation.close()
    # f_training.close()
